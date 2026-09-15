"""Agent tool surface for Kanban checklists (kanban_checklist / kanban_check / kanban_uncheck / create / complete)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="worker-test", assignee="test-worker",
                             checklist=["ai:実装", "machine:pytest"])
        other = kb.create_task(conn, title="other", assignee="someone", checklist=["ai:x"])
        kb.claim_task(conn, tid)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    return tid, other


def test_checklist_view_defaults_to_own_task(worker_env):
    from tools import kanban_tools as kt
    tid, _ = worker_env
    out = kt._handle_checklist({})
    d = json.loads(out)
    assert d["task_id"] == tid
    assert [(i["position"], i["kind"], i["text"], i["done"]) for i in d["items"]] == [
        (1, "ai", "実装", False), (2, "machine", "pytest", False)]
    assert d["progress"]["total"] == 2
    assert "実装" in out  # not \\u-escaped


def test_check_and_uncheck_with_evidence(worker_env):
    from tools import kanban_tools as kt
    d = json.loads(kt._handle_check({"item": 2, "evidence": "job:99"}))
    assert d["ok"] and d["changed"] and d["item"]["done"] and d["item"]["evidence"] == "job:99"
    assert d["progress"]["machine"] == {"total": 1, "done": 1}
    d = json.loads(kt._handle_uncheck({"item_id": d["item"]["id"]}))
    assert d["changed"] and not d["item"]["done"] and d["progress"]["done"] == 0


def test_check_errors_are_tool_errors(worker_env):
    from tools import kanban_tools as kt
    _, other = worker_env
    assert "exactly one" in kt._handle_check({})
    assert "no checklist item #7" in kt._handle_check({"item": 7})
    assert "scoped to task" in kt._handle_check({"task_id": other, "item": 1})


def test_show_includes_progress_and_context(worker_env):
    from tools import kanban_tools as kt
    d = json.loads(kt._handle_show({}))
    assert d["checklist_progress"]["total"] == 2
    assert "## チェックリスト (0/2 済)" in d["worker_context"]


def test_create_with_checklist_and_invalid_checklist(worker_env):
    from hermes_cli import kanban_db_checklist as kbcl
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt
    d = json.loads(kt._handle_create({
        "title": "child", "assignee": "engineer",
        "checklist": [{"text": "build", "kind": "machine"}, {"text": "review"}]}))
    with kbc.connect_closing() as conn:
        assert [(i.kind, i.text) for i in kbcl.list_items(conn, d["task_id"])] == [("machine", "build"), ("ai", "review")]
    err = kt._handle_create({"title": "bad", "assignee": "engineer", "checklist": [{"text": "x", "kind": "robot"}]})
    assert "kind must be one of" in err


def test_complete_reports_unchecked_items(worker_env):
    from tools import kanban_tools as kt
    kt._handle_check({"item": 1})
    d = json.loads(kt._handle_complete({"summary": "done"}))
    assert d["ok"] and d["unchecked_checklist_items"] == 1


def test_checklist_tools_are_registered_and_exposed():
    import tools.kanban_tools  # noqa: F401  (registers)
    from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
    from toolsets import resolve_toolset
    for name in ("kanban_checklist", "kanban_check", "kanban_uncheck"):
        assert name in resolve_toolset("kanban")
        assert name in EXPOSED_TOOLS
