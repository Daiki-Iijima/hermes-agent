"""Block routing and ready-lane PR wait policy."""
from typing import Optional
from hermes_cli import kanban_db as kb


def route_block(
    kind: Optional[str], reason: Optional[str], source_status: str, *,
    prev_kind: Optional[str], prev_recurrences: int,
) -> tuple[str, str, str, tuple, dict]:
    """``(new_status, event_kind, set_sql, params, payload)`` for :func:`block_task`.

    ``dependency`` never enters the human ``blocked`` bucket: it waits in
    ``todo`` for ``recompute_ready``, so a cron never sees a dependency-wait
    as something to "unblock". Every other kind counts unblock-loop
    recurrences: block_task only fires from running/ready (AFTER an unblock
    returned the task to the pool), so a stored ``block_kind`` equal to the
    incoming one means blocked -> unblocked -> re-block for the same cause
    (un-typed None compares equal to a prior un-typed block). Human waits remain
    blocked; only repeated transient failures return to triage for recovery.
    """
    payload = {"reason": reason, "kind": kind, "source_status": source_status}
    if kind == "dependency":
        return "todo", "dependency_wait", "block_kind    = ?", (kind,), payload
    recurrences = prev_recurrences + 1 if prev_kind == kind else 1
    set_sql = "block_kind    = ?,\n                       block_recurrences = ?"
    payload = {"reason": reason, "kind": kind, "recurrences": recurrences, "source_status": source_status}
    if recurrences >= kb.BLOCK_RECURRENCE_LIMIT and kind == "transient":
        payload["limit"] = kb.BLOCK_RECURRENCE_LIMIT
        return "triage", "block_loop_detected", set_sql, (kind, recurrences), payload
    return "blocked", "blocked", set_sql, (kind, recurrences), payload


def active_pr_guard(conn, task_id, latest_run, now, *, window, url_re):
    """Only durable review feedback authorizes continuing an already-linked PR."""
    changes_requested_handoff = False
    if latest_run is not None and latest_run["outcome"] == "changes_requested":
        handoff_event = conn.execute(
            "SELECT 1 FROM task_events "
            "WHERE task_id = ? AND run_id = ? AND kind = 'changes_requested' LIMIT 1",
            (task_id, latest_run["id"]),
        ).fetchone()
        running_owner = conn.execute(
            "SELECT 1 FROM task_runs "
            "WHERE task_id = ? AND ended_at IS NULL AND status = 'running' LIMIT 1",
            (task_id,),
        ).fetchone()
        changes_requested_handoff = handoff_event is not None and running_owner is None
    for comment in conn.execute(
        "SELECT body FROM task_comments WHERE task_id = ? AND created_at >= ?",
        (task_id, now - window),
    ).fetchall():
        if comment["body"] and url_re.search(comment["body"]):
            return None if changes_requested_handoff else "active_pr"
    return None


def persisted_human_wait(kind: Optional[str], recurrences: int) -> bool:
    """Unknown external kinds fail closed; a fresh untyped triage card is not blocked."""
    if kind in {"dependency", "transient"}:
        return False
    return kind is not None or int(recurrences or 0) > 0


def last_respawn_guard_matches(conn, task_id, reason):
    """Deduplicate identical ticks, not changed reasons or intervening progress."""
    event = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    return bool(event is not None and event["kind"] == "respawn_guarded"
                and kb._json_dict(event["payload"]).get("reason") == reason)
