"""Kanban task checklists: storage/migration, library API, CLI, worker context, completion rule."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_checklist as kbcl
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    kb.init_db()
    return home


def _cli(capsys, *argv: str) -> tuple[int, str, str]:
    root = argparse.ArgumentParser(prog="hermes")
    kc.build_parser(root.add_subparsers(dest="cmd"))
    rc = kc.kanban_command(root.parse_args(["kanban", *argv]))
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


def _event_kinds(conn, tid: str) -> list[str]:
    return [e.kind for e in kb.list_events(conn, tid)]


# --- storage / migration ---

def test_existing_board_without_table_is_upgraded_on_connect(tmp_path):
    db_path = tmp_path / "old-board.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(kb.SCHEMA_SQL)
    conn.executescript("DROP INDEX idx_checklist_task; DROP TABLE task_checklist_items;")
    conn.execute("INSERT INTO tasks (id, title, status, created_at) VALUES ('t_old', 'legacy', 'ready', 1)")
    conn.commit()
    conn.close()
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    with kbc.connect_closing(db_path) as migrated:
        tables = {r[0] for r in migrated.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        indexes = {r[0] for r in migrated.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        assert "task_checklist_items" in tables
        assert "idx_checklist_task" in indexes
        kbcl.add_items(migrated, "t_old", ["machine:pytest passes"])
        assert kbcl.progress(migrated, "t_old")["machine"] == {"total": 1, "done": 0}
        assert kb.get_task(migrated, "t_old").title == "legacy"


def test_kind_check_constraint_is_enforced_by_sqlite(kanban_home):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="x")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO task_checklist_items (task_id, position, text, kind, created_at) "
                "VALUES (?, 1, 't', 'human', 0)", (tid,))


# --- library API ---

def test_crud_positions_progress_and_events(kanban_home):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="feature", checklist=["ai:implement", {"text": "run tests", "kind": "machine"}])
        kbcl.add_items(conn, tid, [{"text": "investigate", "kind": "ai"}], position=1, author="alice")
        kbcl.add_items(conn, tid, ["machine:deploy"], author="alice")
        items = kbcl.list_items(conn, tid)
        assert [(i.position, i.text, i.kind) for i in items] == [
            (1, "investigate", "ai"), (2, "implement", "ai"), (3, "run tests", "machine"), (4, "deploy", "machine")]

        item, changed = kbcl.check_item(conn, tid, position=3, done_by="worker-a", evidence="job:42")
        assert changed and item.done and item.done_by == "worker-a" and item.evidence == "job:42"
        kbcl.check_item(conn, tid, item_id=items[0].id, done_by="worker-a")
        assert kbcl.progress(conn, tid) == {
            "total": 4, "done": 2, "ai": {"total": 2, "done": 1}, "machine": {"total": 2, "done": 1}}

        item, changed = kbcl.uncheck_item(conn, tid, position=3, author="alice")
        assert changed and not item.done and item.evidence is None
        assert kbcl.progress(conn, tid)["done"] == 1
        kinds = _event_kinds(conn, tid)
    assert kinds.count("checklist_added") == 3
    assert kinds.count("checklist_checked") == 2
    assert kinds.count("checklist_unchecked") == 1


def test_recheck_without_new_evidence_is_a_noop(kanban_home):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="x", checklist=["a"])
        kbcl.check_item(conn, tid, position=1, done_by="w", evidence="commit:abc")
        _, changed = kbcl.check_item(conn, tid, position=1, done_by="w")
        assert not changed
        item, changed = kbcl.check_item(conn, tid, position=1, done_by="w", evidence="commit:def")
        assert changed and item.evidence == "commit:def"
        _, changed = kbcl.uncheck_item(conn, tid, position=1)
        _, changed_again = kbcl.uncheck_item(conn, tid, position=1)
        assert changed and not changed_again
        assert _event_kinds(conn, tid).count("checklist_checked") == 2


@pytest.mark.parametrize("items, message", [
    ([""], "text is required"),
    (["x" * 301], "exceeds 300"),
    ([{"text": "ok", "kind": "human"}], "kind must be one of"),
    ([f"step {n}" for n in range(31)], "at most 30"),
])
def test_invalid_items_are_rejected(kanban_home, items, message):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="x")
        with pytest.raises(ValueError, match=message):
            kbcl.add_items(conn, tid, items)
        assert kbcl.list_items(conn, tid) == []


def test_add_respects_per_task_cap_across_calls(kanban_home):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="x", checklist=[f"s{n}" for n in range(30)])
        with pytest.raises(ValueError, match="at most 30"):
            kbcl.add_items(conn, tid, ["one more"])


def test_item_reference_validation(kanban_home):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="x", checklist=["a"])
        other = kb.create_task(conn, title="y", checklist=["b"])
        other_id = kbcl.list_items(conn, other)[0].id
        with pytest.raises(ValueError, match="exactly one"):
            kbcl.check_item(conn, tid, position=1, item_id=other_id)
        with pytest.raises(ValueError, match="no checklist item #5"):
            kbcl.check_item(conn, tid, position=5)
        with pytest.raises(ValueError, match="no checklist item id"):
            kbcl.check_item(conn, tid, item_id=other_id)
        with pytest.raises(ValueError, match="evidence exceeds"):
            kbcl.check_item(conn, tid, position=1, evidence="e" * 501)
        with pytest.raises(ValueError, match="unknown task"):
            kbcl.check_item(conn, "t_missing", position=1)


def test_invalid_checklist_on_create_creates_nothing(kanban_home):
    with kbc.connect_closing() as conn:
        with pytest.raises(ValueError):
            kb.create_task(conn, title="bad", checklist=["machine:"])
        assert kb.list_tasks(conn) == []


def test_set_replaces_list_and_keeps_identical_checked_items(kanban_home):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="x", checklist=["ai:design", "machine:build"])
        kbcl.check_item(conn, tid, position=1, done_by="w", evidence="doc")
        kbcl.set_items(conn, tid, ["ai:research", "ai:design", "machine:build --release"], author="pm")
        items = kbcl.list_items(conn, tid)
        assert [(i.position, i.text, i.done) for i in items] == [
            (1, "research", False), (2, "design", True), (3, "build --release", False)]
        assert items[1].evidence == "doc"
        assert kb.list_events(conn, tid)[-1].payload == {"total": 3, "kept_done": 1, "by": "pm"}


def test_progress_for_tasks_is_one_row_per_requested_task(kanban_home):
    with kbc.connect_closing() as conn:
        a = kb.create_task(conn, title="a", checklist=["x", "machine:y"])
        b = kb.create_task(conn, title="b")
        kbcl.check_item(conn, a, position=2)
        got = kbcl.progress_for_tasks(conn, [a, b])
    assert got[a] == {"total": 2, "done": 1, "ai": {"total": 1, "done": 0}, "machine": {"total": 1, "done": 1}}
    assert got[b]["total"] == 0


def test_delete_task_removes_checklist_rows(kanban_home):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="x", checklist=["a", "b"])
        assert kb.delete_task(conn, tid)
        assert conn.execute("SELECT COUNT(*) FROM task_checklist_items").fetchone()[0] == 0


# --- worker context ---

def test_worker_context_renders_checklist_compactly(kanban_home):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="x", body="do it",
                             checklist=["ai:調査する", "machine:pytest を通す", "ai:PR を書く"])
        kbcl.check_item(conn, tid, position=1, evidence="notes.md")
        ctx = kb.build_worker_context(conn, tid)
        bare = kb.create_task(conn, title="no list")
        bare_ctx = kb.build_worker_context(conn, bare)
    assert "## チェックリスト (1/3 済)" in ctx
    assert "- [x] 1. (AI) 調査する — 証跡: notes.md" in ctx
    assert "- [ ] 2. (機械) pytest を通す" in ctx
    assert "kanban_check" in ctx and "最初の未チェック項目" in ctx
    assert ctx.index("## Body") < ctx.index("## チェックリスト")
    assert "チェックリスト" not in bare_ctx


# --- completion rule ---

def test_complete_with_unchecked_items_succeeds_and_notes_remaining(kanban_home):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="x", checklist=["ai:impl", "machine:deploy"])
        kbcl.check_item(conn, tid, position=1)
        assert kb.complete_task(conn, tid, summary="done-ish")
        assert kb.get_task(conn, tid).status == "done"
        event = next(e for e in kb.list_events(conn, tid) if e.kind == "checklist_incomplete")
        comments = kb.list_comments(conn, tid)
    assert event.payload["remaining"] == 1
    assert event.payload["items"][0]["text"] == "deploy"
    assert comments[-1].author == "kanban" and "#2 (機械) deploy" in comments[-1].body


def test_complete_with_all_items_checked_adds_no_note(kanban_home):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="x", checklist=["a"])
        kbcl.check_item(conn, tid, position=1)
        assert kb.complete_task(conn, tid, summary="ok")
        assert "checklist_incomplete" not in _event_kinds(conn, tid)
        assert kb.list_comments(conn, tid) == []


# --- CLI ---

def test_cli_create_checklist_and_json_surfaces(kanban_home, capsys):
    rc, out, _ = _cli(capsys, "create", "cli card", "--checklist", "ai:実装", "--checklist",
                      "machine:npm test", "--json")
    assert rc == 0
    tid = json.loads(out)["id"]

    rc, out, _ = _cli(capsys, "checklist", tid, "--json")
    payload = json.loads(out)
    assert rc == 0 and payload["task_id"] == tid
    assert [(i["position"], i["kind"], i["text"], i["done"]) for i in payload["items"]] == [
        (1, "ai", "実装", False), (2, "machine", "npm test", False)]
    assert payload["progress"]["total"] == 2

    rc, out, _ = _cli(capsys, "checklist", "check", tid, "2", "--evidence", "job:7", "--author", "bob", "--json")
    item = json.loads(out)["items"][1]
    assert rc == 0 and item["done"] and item["evidence"] == "job:7" and item["done_by"] == "bob"

    rc, out, _ = _cli(capsys, "show", tid, "--json")
    shown = json.loads(out)
    assert shown["checklist_progress"]["machine"] == {"total": 1, "done": 1}
    assert len(shown["checklist"]) == 2

    rc, out, _ = _cli(capsys, "list", "--json")
    row = next(r for r in json.loads(out) if r["id"] == tid)
    assert row["checklist_progress"]["done"] == 1

    rc, out, _ = _cli(capsys, "list")
    assert rc == 0 and tid in out


def test_cli_add_uncheck_set_and_item_id(kanban_home, capsys):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="x", checklist=["a"])
    assert _cli(capsys, "checklist", "add", tid, "build", "it", "--kind", "machine")[0] == 0
    with kbc.connect_closing() as conn:
        second = kbcl.list_items(conn, tid)[1]
    assert (second.text, second.kind) == ("build it", "machine")

    rc, out, _ = _cli(capsys, "checklist", "check", tid, f"id:{second.id}")
    assert rc == 0 and "Checked #2" in out and "[x]  2. (機械) build it" in out
    rc, out, _ = _cli(capsys, "checklist", "uncheck", tid, "2")
    assert rc == 0 and "Unchecked #2" in out

    rc, _, _ = _cli(capsys, "checklist", "set", tid, "--item", "machine:deploy", "--item", "ai:verify")
    with kbc.connect_closing() as conn:
        assert [(i.kind, i.text) for i in kbcl.list_items(conn, tid)] == [("machine", "deploy"), ("ai", "verify")]
    assert rc == 0


def test_cli_checklist_errors(kanban_home, capsys):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="x", checklist=["a"])
    rc, _, err = _cli(capsys, "checklist", "check", tid, "9")
    assert rc == 1 and "no checklist item #9" in err
    rc, _, err = _cli(capsys, "checklist", "check", tid, "second")
    assert rc == 1 and "id:<item_id>" in err
    rc, _, err = _cli(capsys, "checklist", "set", tid)
    assert rc == 2 and "--item" in err
    rc, _, err = _cli(capsys, "checklist", "t_nope")
    assert rc == 1 and "no such task" in err
    rc, _, err = _cli(capsys, "create", "bad", "--checklist", "machine:")
    assert rc == 1 and "text is required" in err


def test_cli_worker_cannot_edit_another_tasks_checklist(kanban_home, capsys, monkeypatch):
    with kbc.connect_closing() as conn:
        mine = kb.create_task(conn, title="mine", checklist=["a"])
        other = kb.create_task(conn, title="other", checklist=["b"])
    monkeypatch.setenv("HERMES_KANBAN_TASK", mine)
    rc, _, err = _cli(capsys, "checklist", "check", other, "1")
    assert rc == 1 and "scoped to task" in err
    assert _cli(capsys, "checklist", other)[0] == 0  # reading stays allowed


def test_cli_complete_reports_unchecked_items(kanban_home, capsys):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="x", checklist=["a", "machine:b"])
    rc, out, _ = _cli(capsys, "complete", tid, "--result", "done")
    assert rc == 0 and "2 checklist item(s) left unchecked" in out
