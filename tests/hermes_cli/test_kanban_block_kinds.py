"""Tests for typed block reasons + the unblock-loop breaker.

Covers the built-in fix for the kanban "blocked loop" — a worker blocks a
task, a cron unblocks it, the worker re-blocks for the same reason, repeat
forever. The fix gives ``block_task`` a typed ``kind`` and a persistent
``block_recurrences`` counter:

* ``dependency`` blocks route to ``todo`` (parent-gated, auto-resumed) and
  never enter the human ``blocked`` bucket a cron would keep unblocking.
* ``needs_input`` / ``capability`` / un-typed blocks land in ``blocked`` and
  remain explicit human waits even when ``block_recurrences`` reaches the limit.
* Repeated ``transient`` blockers route to ``triage`` for recovery.
* ``unblock_task`` deliberately does NOT reset ``block_recurrences`` (the
  amnesia that let the loop run unbounded).
* A successful ``complete_task`` resets the loop memory.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hermes_cli import kanban_decompose as decomp
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_specify as spec


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task(conn, title="t"):
    """Create a task and drive it to ``running`` so block_task can act."""
    tid = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    claimed = kb.claim_task(conn, tid, claimer="worker")
    assert claimed is not None
    return tid


def _make_running_again(conn, tid):
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None


# ---------------------------------------------------------------------------
# Loop breaker
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["needs_input", "capability", None])
def test_repeated_human_blocker_stays_blocked_and_can_be_explicitly_unblocked(
    kanban_home: Path, kind: str | None,
) -> None:
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        assert kb.block_task(conn, tid, reason="human decision", kind=kind)
        assert kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)

        assert kb.block_task(conn, tid, reason="human decision", kind=kind)
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "blocked"
        assert task.block_recurrences == kb.BLOCK_RECURRENCE_LIMIT
        assert not [e for e in kb.list_events(conn, tid) if e.kind == "block_loop_detected"]

        assert kb.unblock_task(conn, tid)
        assert kb.get_task(conn, tid).status == "ready"
        assert len([e for e in kb.list_events(conn, tid) if e.kind == "blocked"]) == 2


def test_human_block_triage_rows_are_not_aux_work_but_transient_triage_is(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with kbc.connect_closing() as conn:
        ordinary = kb.create_task(conn, title="rough idea", triage=True)

        transient = _running_task(conn, title="flaky")
        assert kb.block_task(conn, transient, reason="retry later", kind="transient")
        assert kb.unblock_task(conn, transient)
        _make_running_again(conn, transient)
        assert kb.block_task(conn, transient, reason="retry later", kind="transient")
        assert kb.get_task(conn, transient).status == "triage"

        human_ids = []
        for index, kind in enumerate(
            ("needs_input", "capability", "escalation", "unknown", None),
        ):
            tid = kb.create_task(conn, title=f"legacy human wait {index}", triage=True)
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET block_kind = ?, block_recurrences = ? WHERE id = ?",
                    (kind, kb.BLOCK_RECURRENCE_LIMIT, tid),
                )
                kb._append_event(
                    conn,
                    tid,
                    "block_loop_detected",
                    {"kind": kind, "recurrences": kb.BLOCK_RECURRENCE_LIMIT},
                )
            kb.add_comment(conn, tid, author="worker", body="please decompose this wait")
            human_ids.append(tid)

    assert set(spec.list_triage_ids()) == {ordinary, transient}
    assert set(decomp.list_triage_ids()) == {ordinary, transient}

    spec_aux = MagicMock(side_effect=AssertionError("specifier must not run"))
    decomp_aux = MagicMock(side_effect=AssertionError("decomposer must not run"))
    monkeypatch.setattr(spec, "_call_aux", spec_aux)
    monkeypatch.setattr(decomp, "_call_aux", decomp_aux)
    for tid in human_ids:
        assert spec.specify_task(tid).ok is False
        assert decomp.decompose_task(tid).ok is False
    spec_aux.assert_not_called()
    decomp_aux.assert_not_called()

    with kbc.connect_closing() as conn:
        legacy_id = human_ids[0]
        assert kb.unblock_task(conn, legacy_id)
        assert kb.get_task(conn, legacy_id).status == "ready"
        kinds = [event.kind for event in kb.list_events(conn, legacy_id)]
        assert "block_loop_detected" in kinds
        assert kinds[-1] == "unblocked"










def test_block_loop_detected_event_emitted(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="x", kind="transient")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason="x", kind="transient")
        events = [e for e in kb.list_events(conn, tid)
                  if e.kind == "block_loop_detected"]
        assert events, "expected a block_loop_detected event"
        payload = events[-1].payload or {}
        assert payload.get("recurrences") == 2
        assert payload.get("kind") == "transient"


# ---------------------------------------------------------------------------
# Dependency routing
# ---------------------------------------------------------------------------


def test_dependency_then_parent_done_promotes(kanban_home: Path) -> None:
    """A dependency-parked child becomes ready once its parent completes."""
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        kb.block_task(conn, child, reason="wait", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        # Finish the parent, then let recompute_ready run.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"


# ---------------------------------------------------------------------------
# Completion resets loop memory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Validation + back-compat
# ---------------------------------------------------------------------------
