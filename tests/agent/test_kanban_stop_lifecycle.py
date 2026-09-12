"""Run ownership, handoff, and read-only stop-gate contracts against real SQLite."""

import hashlib
import json
import sqlite3
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.kanban_stop import build_kanban_stop_nudge
from agent.turn_stop_gates import apply_stop_gates
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from tools import kanban_tools  # noqa: F401 — register real handlers
from tools.registry import registry


@pytest.fixture
def worker(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "guard-test")
    monkeypatch.setenv("HERMES_PROFILE", "coder")
    monkeypatch.delenv("HERMES_KANBAN_STOP_NUDGE", raising=False)
    # Match dispatcher DB pinning, including URI quoting of filesystem paths.
    path = home / "pinned # board.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(path))
    (home / "profiles" / "reviewer").mkdir(parents=True)
    conn = kbc.connect(path)
    try:
        tid = kb.create_task(conn, title="Implement", assignee="coder")
        children = [kb.create_task(conn, title=f"Child {i}", assignee="coder", parents=[tid])
                    for i in range(2)]
        claimed = kb.claim_task(conn, tid, claimer="coder:guard-test")
        assert claimed is not None
        monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
        yield SimpleNamespace(conn=conn, tid=tid, run_id=claimed.current_run_id,
                              children=children, path=path, home=home)
    finally:
        conn.close()


def _tool_messages(name, args):
    result = registry.dispatch(name, args)
    payload = json.loads(result)
    messages = [{"role": "assistant", "tool_calls": [{
        "id": "lifecycle-1", "function": {"name": name, "arguments": json.dumps(args)},
    }]}, {"role": "tool", "tool_call_id": "lifecycle-1", "content": result}]
    return payload, messages


def _snapshot(worker):
    children = [dict(worker.conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone())
                for tid in worker.children]
    return ("\n".join(worker.conn.iterdump()).encode(),
            json.dumps(children, sort_keys=True, separators=(",", ":")).encode())


def _gate(messages, attempts=0):
    agent = SimpleNamespace(_kanban_stop_nudges=attempts, _emit_status=lambda _: None,
                            _interim_content_was_streamed=lambda _: False)
    verdict = apply_stop_gates(
        agent, {"role": "assistant", "content": "Work handed off."},
        final_response="Work handed off.", messages=messages, conversation_history=[],
        pending_verification_response=None, pending_verification_response_previewed=False,
    )
    return agent, verdict


@pytest.mark.parametrize("handoff", ["review", "changes"])
@pytest.mark.parametrize("transcript", ["success", "compacted", "failed"])
def test_handoff_and_stale_writes_preserve_successor_and_children(
    worker, monkeypatch, record_property, handoff, transcript,
):
    result, messages = _tool_messages("kanban_request_review", {
        "summary": "Implementation ready", "reviewer": "reviewer",
    })
    assert result["ok"] is True
    if handoff == "changes":
        reviewer = kb.claim_review_task(worker.conn, worker.tid, claimer="reviewer:guard-test")
        assert reviewer is not None
        worker.run_id = reviewer.current_run_id
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(worker.run_id))
        result, messages = _tool_messages("kanban_request_changes", {"reason": "Add coverage"})
        assert result["ok"] is True
    outcome = "review_requested" if handoff == "review" else "changes_requested"
    claim_next = kb.claim_review_task if handoff == "review" else kb.claim_task
    successor = claim_next(worker.conn, worker.tid, claimer="successor:guard-test")
    assert successor is not None
    assert successor.current_run_id != worker.run_id
    old = kb.get_run(worker.conn, worker.run_id)
    assert old.outcome == outcome and old.ended_at is not None
    assert old.error is None
    assert kb.goal_run_status(worker.conn, worker.tid, worker.run_id) in {"review", "changes_requested"}
    before, children_before = _snapshot(worker)
    # Real registry calls with the OLD dispatcher run still pinned.
    rejected = []
    for name, args in [("kanban_complete", {"summary": "stale completion"}),
                       ("kanban_block", {"reason": "stale blocker"})]:
        failure, failed_messages = _tool_messages(name, args)
        assert failure.get("error") and failure.get("ok") is not True
        assert _snapshot(worker) == (before, children_before)
        rejected.append(failure)
    if transcript == "compacted":
        messages = []
    elif transcript == "failed":
        messages = failed_messages
    original_messages = list(messages)
    agent, verdict = _gate(messages)
    after, children_after = _snapshot(worker)
    evidence = {
        "handoff": handoff, "transcript": transcript, "old_run_id": worker.run_id,
        "old_outcome": old.outcome, "successor_run_id": successor.current_run_id,
        "nudges": agent._kanban_stop_nudges, "stale_rejections": rejected,
        "db_before_sha256": hashlib.sha256(before).hexdigest(),
        "db_after_sha256": hashlib.sha256(after).hexdigest(),
        "children_before": children_before.decode(), "children_after": children_after.decode(),
    }
    record_property("db_state_evidence", json.dumps(evidence, sort_keys=True))
    print(json.dumps(evidence, sort_keys=True))
    assert after == before
    assert children_after == children_before
    assert kb.get_run(worker.conn, worker.run_id) == old
    current = kb.get_task(worker.conn, worker.tid)
    assert current.status == "running" and current.current_run_id == successor.current_run_id
    assert kb.get_run(worker.conn, successor.current_run_id).outcome is None
    assert agent._kanban_stop_nudges == 0
    assert verdict.continue_turn is False and verdict.final_response == "Work handed off."
    assert messages == original_messages


@pytest.mark.parametrize("case", [
    "owned", "second_attempt", "failed_complete", "failed_block", "failed_review", "failed_changes",
    "stale_success", "disabled", "exhausted", "custom_max", "missing_task",
    "unknown_task", "task_override", "missing_run", "invalid_run", "zero_run", "negative_run",
    "huge_run", "unknown_run", "other_task_run", "missing_board", "invalid_board",
    "missing_db", "legacy_db", "corrupt_db", "locked_db", "ended_run", "lost_pointer",
    "completed", "blocked", "review", "changes", "descendant", "foreign_context",
])
def test_only_verified_active_owner_is_nudged(worker, monkeypatch, case):
    messages = []
    options = {}
    should_nudge = case in {
        "owned", "second_attempt", "failed_complete", "failed_block", "failed_review", "failed_changes", "stale_success",
    }
    if case.startswith("failed_"):
        name = {"failed_complete": "kanban_complete", "failed_block": "kanban_block",
                "failed_review": "kanban_request_review", "failed_changes": "kanban_request_changes"}[case]
        args = {"reason": "Not a review run"} if case == "failed_changes" else {}
        result, messages = _tool_messages(name, args)
        assert result.get("error")
    if case == "stale_success":
        messages = [{"role": "assistant", "tool_calls": [{"id": "old", "function": {
            "name": "kanban_complete", "arguments": "{}",
        }}]}, {"role": "tool", "tool_call_id": "old", "content": json.dumps({
            "ok": True, "task_id": worker.tid, "run_id": worker.run_id,
        })}]
    env_cases = {
        "missing_task": ("HERMES_KANBAN_TASK", ""), "unknown_task": ("HERMES_KANBAN_TASK", "t_unknown"),
        "missing_run": ("HERMES_KANBAN_RUN_ID", ""), "invalid_run": ("HERMES_KANBAN_RUN_ID", "broken"),
        "zero_run": ("HERMES_KANBAN_RUN_ID", "0"), "negative_run": ("HERMES_KANBAN_RUN_ID", "-1"),
        "huge_run": ("HERMES_KANBAN_RUN_ID", str(2**100)),
        "unknown_run": ("HERMES_KANBAN_RUN_ID", str(worker.run_id + 999)),
        "missing_board": ("HERMES_KANBAN_BOARD", ""), "invalid_board": ("HERMES_KANBAN_BOARD", "../bad"),
        "disabled": ("HERMES_KANBAN_STOP_NUDGE", "off"),
        "descendant": ("HERMES_DELEGATED_CHILD_CONTEXT", "1"),
    }
    if case in env_cases:
        monkeypatch.setenv(*env_cases[case])
    if case == "other_task_run":
        other = kb.create_task(worker.conn, title="Other", assignee="coder")
        claim = kb.claim_task(worker.conn, other)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claim.current_run_id))
    if case == "task_override":
        options["task_id"] = worker.children[0]
    if case == "second_attempt":
        options["attempts"] = 1
    if case in {"exhausted", "custom_max"}:
        options["attempts"] = 2 if case == "exhausted" else 1
        if case == "custom_max":
            options["max_attempts"] = 1
    if case in {"missing_db", "legacy_db", "corrupt_db"}:
        path = worker.home / "absent" / "kanban.db" if case == "missing_db" else worker.home / "unreadable.db"
        if case == "legacy_db":
            with sqlite3.connect(path) as conn:
                conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, status TEXT)")
        if case == "corrupt_db":
            path.write_bytes(b"not a sqlite database")
        monkeypatch.setenv("HERMES_KANBAN_DB", str(path))
    if case == "ended_run":
        worker.conn.execute("UPDATE task_runs SET ended_at = 1 WHERE id = ?", (worker.run_id,))
    if case == "lost_pointer":
        worker.conn.execute("UPDATE tasks SET current_run_id = NULL WHERE id = ?", (worker.tid,))
    if case in {"completed", "blocked", "review", "changes"}:
        if case in {"review", "changes"}:
            assert kb.request_review(worker.conn, worker.tid, reviewer="reviewer", expected_run_id=worker.run_id)
            if case == "changes":
                claimed = kb.claim_review_task(worker.conn, worker.tid)
                worker.run_id = claimed.current_run_id
                monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(worker.run_id))
                assert kb.request_changes(worker.conn, worker.tid, reason="Revise", expected_run_id=worker.run_id)[0]
        elif case == "completed":
            assert kb.complete_task(worker.conn, worker.tid, expected_run_id=worker.run_id)
        else:
            assert kb.block_task(worker.conn, worker.tid, reason="External input", expected_run_id=worker.run_id)
    # No connections remain when byte-probing files (POSIX lock safety).
    before = _snapshot(worker)
    worker.conn.close()
    def durable_files():
        # SQLite may create/update its WAL reader-coordination sidecars even in
        # mode=ro. Check durable DB/schema/config bytes, not shared lock memory.
        return {str(p.relative_to(worker.home)): p.read_bytes()
                for p in worker.home.rglob("*")
                if p.is_file() and not p.name.endswith(("-wal", "-shm"))}

    files_before = durable_files()
    lock = None
    if case == "locked_db":
        lock = sqlite3.connect(worker.path)
        lock.execute("PRAGMA journal_mode=DELETE")
        lock.execute("BEGIN EXCLUSIVE")
    from agent.delegation_context import non_dispatcher_owned_context
    try:
        with non_dispatcher_owned_context() if case == "foreign_context" else nullcontext():
            nudge = build_kanban_stop_nudge(messages=messages, **options)
    finally:
        if lock is not None:
            lock.rollback()
            lock.close()
    files_after = durable_files()
    if case != "locked_db":
        assert files_after == files_before  # no create, migrate, quarantine, or checkpoint
    with sqlite3.connect(worker.path) as conn:
        conn.row_factory = sqlite3.Row
        worker.conn = conn
        assert _snapshot(worker) == before
    assert (nudge is not None) is should_nudge
    if should_nudge:
        assert worker.tid in nudge and str(worker.run_id) in nudge
        assert all(tool in nudge for tool in (
            "kanban_complete", "kanban_block", "kanban_request_review", "kanban_request_changes",
        ))
        prior_messages = list(messages)
        attempts = options.get("attempts", 0)
        agent, verdict = _gate(messages, attempts)
        assert agent._kanban_stop_nudges == attempts + 1
        assert verdict.continue_turn and verdict.final_response is None
        assert verdict.pending_verification_response == "Work handed off."
        assert messages[:-2] == prior_messages
        assert [message["role"] for message in messages[-2:]] == ["assistant", "user"]
