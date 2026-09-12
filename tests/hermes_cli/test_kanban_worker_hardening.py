"""Worker ownership must precede goal turns and durable completion validation."""
from contextlib import ExitStack
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest


@pytest.fixture
def board(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for key, value in {
        "HERMES_HOME": str(home), "HERMES_KANBAN_HOME": str(home),
        "HERMES_KANBAN_BOARD": "hardening", "HERMES_PROFILE": "implementer",
        "HERMES_KANBAN_GOAL_MODE": "1", "HERMES_VERIFY_ON_STOP": "0",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    # CLI startup installs log handlers; measure the loop after startup finishes.
    import cli  # noqa: F401

    (home / "profiles/reviewer").mkdir(parents=True)
    path = kb.kanban_db_path(board="hardening")
    conn = kbc.connect(path)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(path))
    tid = kb.create_task(conn, title="Review ownership", assignee="implementer",
                         workspace_kind="scratch", goal_max_turns=3)
    child = kb.create_task(conn, title="Dependent", assignee="implementer",
                           parents=[tid], workspace_kind="scratch")
    run = kb.claim_task(conn, tid, claimer="implementer:first").current_run_id
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run))
    try:
        yield SimpleNamespace(kb=kb, conn=conn, home=home, path=path,
                              tid=tid, child=child, run=run)
    finally:
        conn.close()


def _snapshot(board):
    return "\n".join(board.conn.iterdump()), board.kb.get_task(board.conn, board.child)


def _tool(name, args):
    import tools.kanban_tools  # noqa: F401 - register real tool handlers
    from model_tools import handle_function_call

    return json.loads(handle_function_call(name, args, tool_call_id="hardening-call"))


def _reclaim(board, monkeypatch, handoff):
    assert _tool("kanban_request_review", {"summary": "Ready", "reviewer": "reviewer"})["ok"]
    if handoff == "changes":
        board.run = board.kb.claim_review_task(
            board.conn, board.tid, claimer="reviewer:first").current_run_id
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(board.run))
        assert _tool("kanban_request_changes", {"reason": "Add coverage"})["ok"]
    old = board.kb.get_run(board.conn, board.run)
    assert old.outcome == {"review": "review_requested", "changes": "changes_requested"}[handoff]
    assert old.ended_at is not None and old.error is None
    claim = board.kb.claim_review_task if handoff == "review" else board.kb.claim_task
    successor = claim(board.conn, board.tid, claimer="successor")
    assert successor.id == board.tid and successor.current_run_id != board.run
    return old, board.kb.get_run(board.conn, successor.current_run_id)


def _probe_goal(board, monkeypatch, verdict, on_turn=None, on_judge=None):
    from cli import _run_kanban_goal_loop_q
    from hermes_cli import goals

    calls = {"judge": [], "turn": [], "block": [], "errors": []}
    real_block = board.kb.block_task

    def judge(*args):
        calls["judge"].append(args)
        if on_judge:
            on_judge()
        return verdict, "offline verdict", False, None, False

    def turn(**kwargs):
        calls["turn"].append(kwargs)
        if on_turn:
            on_turn()
        return {"final_response": "Finished", "messages": []}

    def block(*args, **kwargs):
        calls["block"].append(kwargs)
        return real_block(*args, **kwargs)

    monkeypatch.setattr(goals, "judge_goal", judge)
    monkeypatch.setattr(board.kb, "block_task", block)
    cli = SimpleNamespace(agent=SimpleNamespace(run_conversation=turn), conversation_history=[])
    try:
        _run_kanban_goal_loop_q(cli, "First response")
    except Exception as exc:
        calls["errors"].append(repr(exc))
    return calls


@pytest.mark.parametrize("verdict", ["blocked", "continue"])
@pytest.mark.parametrize("case", [
    "missing_run", "empty_run", "bad_run", "float_run", "zero_run", "negative_run",
    "overflow_run", "huge_run", "unknown_run", "max_run", "missing_task", "unknown_task", "wrong_task",
    "missing_board", "empty_board", "invalid_board", "wrong_board", "board_override",
    "missing_db", "wrong_db", "legacy_db", "corrupt_db", "delegated", "child_process",
    "non_dispatcher", "old_review", "old_changes",
])
def test_goal_requires_owned_task_board_and_run(board, monkeypatch, case, verdict):
    from agent import delegation_context as delegation
    from hermes_cli import kanban_db_connect as kbc

    old = successor = None
    with ExitStack() as stack:
        pins = {
            "missing_run": ("HERMES_KANBAN_RUN_ID", None),
            "empty_run": ("HERMES_KANBAN_RUN_ID", " "),
            "bad_run": ("HERMES_KANBAN_RUN_ID", "invalid"),
            "float_run": ("HERMES_KANBAN_RUN_ID", "1.5"),
            "zero_run": ("HERMES_KANBAN_RUN_ID", "0"),
            "negative_run": ("HERMES_KANBAN_RUN_ID", "-1"),
            "overflow_run": ("HERMES_KANBAN_RUN_ID", str(2**63)),
            "huge_run": ("HERMES_KANBAN_RUN_ID", str(2**100)),
            "unknown_run": ("HERMES_KANBAN_RUN_ID", str(board.run + 100)),
            "max_run": ("HERMES_KANBAN_RUN_ID", str(2**63 - 1)),
            "missing_task": ("HERMES_KANBAN_TASK", None),
            "unknown_task": ("HERMES_KANBAN_TASK", "t_unknown"),
            "missing_board": ("HERMES_KANBAN_BOARD", None),
            "empty_board": ("HERMES_KANBAN_BOARD", " "),
            "invalid_board": ("HERMES_KANBAN_BOARD", "../escape"),
            "child_process": ("HERMES_DELEGATED_CHILD_CONTEXT", "1"),
        }
        if case in pins:
            key, value = pins[case]
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)
        if case == "wrong_task":
            other = board.kb.create_task(board.conn, title="Other", workspace_kind="scratch")
            board.kb.claim_task(board.conn, other)
            monkeypatch.setenv("HERMES_KANBAN_TASK", other)
        if case == "wrong_board":
            monkeypatch.delenv("HERMES_KANBAN_DB")
            monkeypatch.setenv("HERMES_KANBAN_BOARD", "nonexistent")
            # A stale slug must not fall back to a valid current board.
            board.kb.set_current_board("hardening")
        if case == "board_override":
            stack.enter_context(board.kb.scoped_current_board("other"))
        if case in {"missing_db", "wrong_db", "legacy_db", "corrupt_db"}:
            path = board.home / "absent-parent/missing.db" if case == "missing_db" else board.home / "other.db"
            if case == "wrong_db":
                conn = stack.enter_context(kbc.connect_closing(path))
                other = board.kb.create_task(conn, title="Other board", workspace_kind="scratch")
                board.kb.claim_task(conn, other)
            if case == "legacy_db":
                conn = sqlite3.connect(path)
                conn.execute("CREATE TABLE tasks (id TEXT, status TEXT)")
                conn.commit()
                conn.close()
            if case == "corrupt_db":
                path.write_bytes(b"not a sqlite database")
            monkeypatch.setenv("HERMES_KANBAN_DB", str(path))
        contexts = {"delegated": delegation.delegated_child_context,
                    "non_dispatcher": delegation.non_dispatcher_owned_context}
        if case in contexts:
            stack.enter_context(contexts[case]())
        if case.startswith("old_"):
            old, successor = _reclaim(board, monkeypatch, case.removeprefix("old_"))

        before = _snapshot(board)
        paths = {str(p.relative_to(board.home)) for p in board.home.rglob("*")
                 if not p.name.endswith(("-wal", "-shm"))}
        calls = _probe_goal(board, monkeypatch, verdict)
        assert calls == {"judge": [], "turn": [], "block": [], "errors": []}
        assert _snapshot(board) == before
        assert {str(p.relative_to(board.home)) for p in board.home.rglob("*")
                if not p.name.endswith(("-wal", "-shm"))} == paths
        if old:
            assert board.kb.get_run(board.conn, old.id) == old
            assert board.kb.get_run(board.conn, successor.id) == successor
            assert successor.status == board.kb.get_task(board.conn, board.tid).status == "running"


@pytest.mark.parametrize("mode", ["block", "turn", "custom_db", "board_only", "handoff_block", "handoff_continue"])
def test_goal_preserves_active_owner_and_pins_callbacks(board, monkeypatch, mode):
    if mode == "board_only":
        monkeypatch.delenv("HERMES_KANBAN_DB")
    if mode == "custom_db":
        # Dispatcher DB pins may be outside the canonical board directory.
        path = board.home / "board #1?.db"
        conn = sqlite3.connect(path)
        board.conn.backup(conn)
        conn.close()
        board.conn.close()
        from hermes_cli.kanban_db_connect import connect
        board.conn = connect(path)
        board.path = path
        monkeypatch.setenv("HERMES_KANBAN_DB", str(path))
    if mode.startswith("handoff_"):
        snapshots = []
        def handoff():
            _reclaim(board, monkeypatch, "review")
            snapshots.append(_snapshot(board))
        calls = _probe_goal(board, monkeypatch, "blocked" if mode == "handoff_block" else "continue", on_judge=handoff)
        assert len(calls["judge"]) == 1 and not calls["turn"] and not calls["errors"]
        assert all(c["expected_run_id"] == board.run for c in calls["block"])
        assert _snapshot(board) == snapshots[0]
        return

    def finish():
        assert board.kb.complete_task(board.conn, board.tid, expected_run_id=board.run)

    verdict = "continue" if mode == "turn" else "blocked"
    calls = _probe_goal(board, monkeypatch, verdict, on_turn=finish)
    assert not calls["errors"] and len(calls["judge"]) == 1
    assert len(calls["turn"]) == (mode == "turn")
    assert len(calls["block"]) == (mode != "turn")
    assert all(c["expected_run_id"] == board.run for c in calls["block"])
    assert board.kb.get_task(board.conn, board.tid).status == ("done" if mode == "turn" else "blocked")


@pytest.mark.parametrize("handoff", ["review", "changes"])
@pytest.mark.parametrize("payload", ["ordinary", "invalid_cards", "prose", "acceptance"])
def test_stale_completion_and_block_leave_every_durable_table_unchanged(board, monkeypatch, handoff, payload):
    child_before = _snapshot(board)[1]
    old, successor = _reclaim(board, monkeypatch, handoff)
    args = {"summary": "Stale completion"}
    if payload == "invalid_cards":
        args["created_cards"] = ["t_nonexistent"]
    if payload == "prose":
        args["summary"] = "Created t_deadbeef12345678"
    if payload == "acceptance":
        board.conn.execute("UPDATE tasks SET completion_contract=? WHERE id=?", ("acme/repo", board.tid))
        args["metadata"] = {"published_pr": "https://github.com/acme/repo/pull/7"}
        from hermes_cli import kanban_pr_acceptance_store as acceptance
        monkeypatch.setattr(acceptance, "collect_acceptance", lambda *a: pytest.fail("Stale acceptance must not run"))
    before = _snapshot(board)
    assert before[1] == child_before
    for name, values in [("kanban_complete", args), ("kanban_block", {"reason": "Stale blocker"})]:
        failure = _tool(name, values)
        assert failure.get("error") and failure.get("ok") is not True
        assert _snapshot(board) == before
        assert board.kb.get_run(board.conn, old.id) == old
        assert board.kb.get_run(board.conn, successor.id) == successor
        assert board.kb.get_task(board.conn, board.tid).current_run_id == successor.id
        assert successor.status == board.kb.get_task(board.conn, board.tid).status == "running"
        assert successor.ended_at is None and successor.outcome is None
    # A stale caller is rejected before the card-validation helper is entered.
    monkeypatch.setattr(board.kb, "_gate_created_cards", lambda *a, **kw: pytest.fail("Stale validation ran"))
    assert board.kb.complete_task(board.conn, board.tid, created_cards=["t_nonexistent"],
                                  expected_run_id=old.id) is False
    assert _snapshot(board) == before


@pytest.mark.parametrize("owner", ["active", "human", "race"])
def test_completion_validation_audit_is_owned_and_atomic(board, monkeypatch, owner):
    before = _snapshot(board)
    kwargs = {} if owner == "human" else {"expected_run_id": board.run}
    if owner == "race":
        # Try a real second SQLite writer while the card iterable is consumed.
        # Validation must hold the same ownership transaction as its audit.
        locked = []
        rival = sqlite3.connect(board.path, timeout=0, isolation_level=None)
        def cards():
            try:
                rival.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as exc:
                locked.append("locked" in str(exc).lower())
            else:
                rival.rollback()
                locked.append(False)
            yield "t_nonexistent"
        created_cards = cards()
    else:
        created_cards = ["t_nonexistent"]
    try:
        with pytest.raises(board.kb.HallucinatedCardsError):
            board.kb.complete_task(board.conn, board.tid, summary="Invalid cards",
                                   created_cards=created_cards, **kwargs)
    finally:
        if owner == "race":
            rival.close()
    if owner == "race":
        assert locked == [True], "ownership check and durable validation must not have a writer race"
    events = board.kb.list_events(board.conn, board.tid)
    audit = events[-1]
    assert audit.kind == "completion_blocked_hallucination"
    assert audit.payload["phantom_cards"] == ["t_nonexistent"]
    assert _snapshot(board)[1] == before[1]
    assert board.kb.get_task(board.conn, board.tid).status == "running"
    assert board.kb.get_run(board.conn, board.run).ended_at is None
    assert not board.conn.in_transaction
    assert board.kb.complete_task(board.conn, board.tid, summary="Corrected", created_cards=[], **kwargs)
    assert board.kb.get_task(board.conn, board.tid).status == "done"
