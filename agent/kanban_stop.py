"""Bounded turn-end nudges for a dispatcher-owned, still-active Kanban run.

The persisted run and task pointer are authoritative, including review handoffs.
Missing identity or unreadable state means no nudge: transcript evidence alone
must never authorize another lifecycle write.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import closing
from typing import Any, Iterable, NamedTuple, Optional


logger = logging.getLogger(__name__)
_TERMINAL_KANBAN_TOOLS = frozenset({
    "kanban_complete", "kanban_block", "kanban_request_review", "kanban_request_changes",
})
_DEFAULT_MAX_ATTEMPTS = 2


class _WorkerRun(NamedTuple):
    board: str
    task_id: str
    run_id: int


def kanban_stop_nudge_enabled() -> bool:
    """Only dispatcher workers inherit authority from the pinned environment."""
    from agent.delegation_context import is_dispatcher_owned_worker_context

    if (os.environ.get("HERMES_KANBAN_STOP_NUDGE") or "").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool((os.environ.get("HERMES_KANBAN_TASK") or "").strip()) and is_dispatcher_owned_worker_context()


def _worker_run(task_id: Optional[str] = None) -> Optional[_WorkerRun]:
    from hermes_cli.kanban_db import _normalize_board_slug

    tid = (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    try:
        board = _normalize_board_slug(os.environ.get("HERMES_KANBAN_BOARD"))
        run_id = int(os.environ.get("HERMES_KANBAN_RUN_ID") or "")
    except ValueError:
        logger.debug("kanban stop guard skipped: invalid dispatcher identity")
        return None
    if not tid or not board or not 0 < run_id < 2**63 or (task_id is not None and task_id != tid):
        logger.debug("kanban stop guard skipped: missing or mismatched dispatcher identity")
        return None
    return _WorkerRun(board, tid, run_id)


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _json_object(value: Any) -> Optional[dict]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return value if isinstance(value, dict) else None


def session_called_kanban_terminal(messages: Iterable[dict] | None) -> bool:
    """Match successful results to calls for the pinned task, board and run.

    This is transcript evidence, not a substitute for persisted ownership. A
    rejected, pending, unrelated, or compacted-away call proves no handoff.
    """
    worker = _worker_run()
    if worker is None:
        return False
    calls = {}
    for msg in messages or ():
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or ():
                fn = _field(tc, "function", tc)
                name = _field(fn, "name")
                args = _json_object(_field(fn, "arguments", "{}"))
                call_id = _field(tc, "id")
                if call_id and name in _TERMINAL_KANBAN_TOOLS and args is not None:
                    if (args.get("task_id") or worker.task_id) != worker.task_id:
                        continue
                    if (args.get("board") or worker.board) != worker.board:
                        continue
                    calls[call_id] = name
        elif msg.get("role") == "tool":
            name = calls.pop(msg.get("tool_call_id"), None)
            result = _json_object(msg.get("content"))
            if name and result and (not msg.get("name") or msg["name"] == name):
                if (result.get("ok") is True and not result.get("error")
                        and result.get("task_id") == worker.task_id
                        and result.get("run_id") == worker.run_id
                        and (result.get("board") or worker.board) == worker.board):
                    return True
    return False


def _owns_active_run(worker: _WorkerRun) -> bool:
    from hermes_cli.kanban_db import kanban_db_path
    from hermes_cli.sqlite_safe_read import connect_tracked

    # Never use kanban_db_connect.connect here: it creates/migrates/repairs.
    # Explicit board avoids falling through to the currently selected board;
    # the dispatcher's HERMES_KANBAN_DB pin retains its canonical precedence.
    path = kanban_db_path(board=worker.board)
    with closing(connect_tracked(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.1)) as conn:
        # One SELECT gives a consistent snapshot of BOTH the run and ownership.
        # All ended outcomes (including review/changes requested) and superseded
        # runs stop here, even if the transcript is missing or says otherwise.
        return conn.execute(
            "SELECT 1 FROM tasks t JOIN task_runs r ON r.task_id = t.id "
            "WHERE t.id = ? AND r.id = ? AND t.current_run_id = r.id "
            "AND t.status = 'running' AND r.status = 'running' "
            "AND r.ended_at IS NULL AND r.outcome IS NULL",
            (worker.task_id, worker.run_id),
        ).fetchone() is not None


def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Nudge only a verified active owner, within the existing retry budget.

    Unknown/malformed identity, absent/legacy/corrupt/locked DBs, terminal runs,
    and lost ownership all return None without creating or modifying board state.
    """
    if not kanban_stop_nudge_enabled() or attempts >= max_attempts:
        return None
    worker = _worker_run(task_id)
    if worker is None:
        return None
    try:
        if not _owns_active_run(worker):
            return None
    except (sqlite3.Error, OSError, ValueError):
        logger.debug("kanban stop guard skipped: persisted ownership unavailable", exc_info=True)
        return None

    prior_result = (
        "A lifecycle call reported success, but persisted state still records this run as active. "
        if session_called_kanban_terminal(messages) else ""
    )
    return (
        "[System: You are a Hermes kanban worker. A plain-text reply is not a "
        "terminal outcome for your run.\n\n"
        f"The board `{worker.board}` currently records task `{worker.task_id}` "
        f"as `running` under your run `{worker.run_id}`. "
        f"{prior_result}Before taking further action, check that this run still owns the task. "
        "If it has ended or another run owns it, stop without changing the task.\n\n"
        "For your active run, finish any remaining deliverable, then use the appropriate "
        "lifecycle tool: `kanban_complete(summary=..., artifacts=[...])` for finished work, "
        "`kanban_request_review(summary=...)` to hand off implementation, "
        "`kanban_request_changes(reason=...)` to return review feedback, or "
        "`kanban_block(reason=...)` for a blocker. Check the tool result; a rejected call "
        "does not end the run. Exiting an active run without a terminal outcome causes "
        "a protocol violation.]"
    )


__all__ = ["build_kanban_stop_nudge", "kanban_stop_nudge_enabled", "session_called_kanban_terminal"]
