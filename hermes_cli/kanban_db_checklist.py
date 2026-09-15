"""Per-task checklists: ordered, verifiable steps tagged ``ai`` or ``machine``.

Rows live in ``task_checklist_items`` (see ``SCHEMA_SQL``). Every mutation
emits a ``checklist_*`` task event inside the same write transaction so the
Board activity feed and notify cursors see it like any other change.
Positions are 1-based and kept dense (1..N) by every writer here.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional

from hermes_cli import kanban_db as kb

logger = logging.getLogger(__name__)

CHECKLIST_KINDS = ("ai", "machine")
CHECKLIST_MAX_ITEMS = 30
CHECKLIST_MAX_TEXT = 300
CHECKLIST_MAX_EVIDENCE = 500
_MAX_ACTOR = 100
_KIND_LABELS = {"ai": "AI", "machine": "機械"}


@dataclass
class ChecklistItem:
    id: int
    task_id: str
    position: int
    text: str
    kind: str
    done_at: Optional[int]
    done_by: Optional[str]
    evidence: Optional[str]
    created_at: int

    @property
    def done(self) -> bool:
        return self.done_at is not None

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "done": self.done}


def _item(row: sqlite3.Row) -> ChecklistItem:
    return ChecklistItem(**{k: row[k] for k in ChecklistItem.__dataclass_fields__})


# --- Validation / parsing ---

def _validated(text: Any, kind: Any) -> tuple[str, str]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("checklist item text is required")
    text = " ".join(text.split())
    if len(text) > CHECKLIST_MAX_TEXT:
        raise ValueError(f"checklist item text exceeds {CHECKLIST_MAX_TEXT} chars")
    kind = (kind or "ai").strip().lower() if isinstance(kind, str) or kind is None else kind
    if kind not in CHECKLIST_KINDS:
        raise ValueError(f"checklist item kind must be one of {list(CHECKLIST_KINDS)}, got {kind!r}")
    return text, kind


def parse_item_spec(spec: str, default_kind: str = "ai") -> tuple[str, str]:
    """``"machine:run pytest"`` -> ``("run pytest", "machine")``; no known prefix -> ``default_kind``."""
    if not isinstance(spec, str):
        raise ValueError("checklist item must be a string")
    head, sep, rest = spec.partition(":")
    if sep and head.strip().lower() in CHECKLIST_KINDS:
        return _validated(rest, head)
    return _validated(spec, default_kind)


def normalize_items(items: Optional[Iterable[Any]]) -> list[tuple[str, str]]:
    """Strict: ``"kind:text"`` strings or ``{"text", "kind"}`` dicts -> ``[(text, kind)]``.
    Raises ``ValueError`` on any invalid entry or when over the per-task cap."""
    if items is None:
        return []
    if isinstance(items, (str, dict)):
        items = [items]
    out = []
    for entry in items:
        if isinstance(entry, dict):
            out.append(_validated(entry.get("text"), entry.get("kind")))
        else:
            out.append(parse_item_spec(entry))
    if len(out) > CHECKLIST_MAX_ITEMS:
        raise ValueError(f"a task can have at most {CHECKLIST_MAX_ITEMS} checklist items")
    return out


def lenient_items(raw: Any, *, context: str) -> list[dict[str, str]]:
    """LLM-output variant: skip invalid entries with a warning, never raise.
    Returns ``[{"text", "kind"}]`` capped at ``CHECKLIST_MAX_ITEMS``."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        logger.warning("%s: checklist is not a list (%s); ignoring it", context, type(raw).__name__)
        return []
    kept: list[dict[str, str]] = []
    for idx, entry in enumerate(raw):
        try:
            text, kind = (_validated(entry.get("text"), entry.get("kind")) if isinstance(entry, dict)
                          else parse_item_spec(entry))
        except ValueError as exc:
            logger.warning("%s: skipping checklist[%d]: %s", context, idx, exc)
            continue
        kept.append({"text": text, "kind": kind})
    if len(kept) > CHECKLIST_MAX_ITEMS:
        logger.warning("%s: checklist has %d items; keeping the first %d", context, len(kept), CHECKLIST_MAX_ITEMS)
        kept = kept[:CHECKLIST_MAX_ITEMS]
    return kept


def _actor(value: Optional[str]) -> Optional[str]:
    value = (value or "").strip()
    return value[:_MAX_ACTOR] or None


def _evidence(value: Optional[str]) -> Optional[str]:
    if value is None or not str(value).strip():
        return None
    value = str(value).strip()
    if len(value) > CHECKLIST_MAX_EVIDENCE:
        raise ValueError(f"evidence exceeds {CHECKLIST_MAX_EVIDENCE} chars")
    return value


# --- Reads ---

def list_items(conn: sqlite3.Connection, task_id: str) -> list[ChecklistItem]:
    rows = conn.execute(
        "SELECT * FROM task_checklist_items WHERE task_id = ? ORDER BY position, id", (task_id,),
    ).fetchall()
    return [_item(r) for r in rows]


def _empty_progress() -> dict[str, Any]:
    return {"total": 0, "done": 0, "ai": {"total": 0, "done": 0}, "machine": {"total": 0, "done": 0}}


def progress_for_tasks(conn: sqlite3.Connection, task_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    """``{task_id: progress}`` for every id (zeros when no items) in one grouped query."""
    ids = list(dict.fromkeys(task_ids))
    out = {tid: _empty_progress() for tid in ids}
    for start in range(0, len(ids), 500):  # stay under SQLite's host-parameter limit
        chunk = ids[start:start + 500]
        rows = conn.execute(
            "SELECT task_id, kind, COUNT(*) AS total, COUNT(done_at) AS done "
            "FROM task_checklist_items WHERE task_id IN (" + ",".join("?" * len(chunk)) + ") "
            "GROUP BY task_id, kind", chunk,
        ).fetchall()
        for r in rows:
            p = out[r["task_id"]]
            if r["kind"] in CHECKLIST_KINDS:
                p[r["kind"]] = {"total": r["total"], "done": r["done"]}
            p["total"] += r["total"]
            p["done"] += r["done"]
    return out


def progress(conn: sqlite3.Connection, task_id: str) -> dict[str, Any]:
    return progress_for_tasks(conn, [task_id])[task_id]


# --- Writes ---

def _insert(conn: sqlite3.Connection, task_id: str, items: list[tuple[str, str]], start: int, now: int) -> list[int]:
    ids = []
    for offset, (text, kind) in enumerate(items):
        cur = conn.execute(
            "INSERT INTO task_checklist_items (task_id, position, text, kind, created_at) VALUES (?, ?, ?, ?, ?)",
            (task_id, start + offset, text, kind, now),
        )
        ids.append(int(cur.lastrowid))
    return ids


def insert_initial_items(
    conn: sqlite3.Connection, task_id: str, items: list[tuple[str, str]], *, by: Optional[str],
) -> None:
    """Raw insert for a brand-new task inside the caller's write txn (create/decompose)."""
    if not items:
        return
    _insert(conn, task_id, items, 1, int(time.time()))
    kb._append_event(conn, task_id, "checklist_added", {"count": len(items), "total": len(items), "by": _actor(by)})


def add_items(
    conn: sqlite3.Connection, task_id: str, items: Iterable[Any], *,
    position: Optional[int] = None, author: Optional[str] = None,
) -> list[int]:
    """Append (or insert before 1-based ``position``) items; returns the new ids."""
    parsed = normalize_items(items)
    if not parsed:
        raise ValueError("no checklist items given")
    with kb.write_txn(conn, allow_nested=True):
        kb._require_task(conn, task_id)
        count = conn.execute(
            "SELECT COUNT(*) FROM task_checklist_items WHERE task_id = ?", (task_id,),
        ).fetchone()[0]
        if count + len(parsed) > CHECKLIST_MAX_ITEMS:
            raise ValueError(f"a task can have at most {CHECKLIST_MAX_ITEMS} checklist items (has {count})")
        start = count + 1 if position is None else int(position)
        if not 1 <= start <= count + 1:
            raise ValueError(f"position must be between 1 and {count + 1}")
        conn.execute(
            "UPDATE task_checklist_items SET position = position + ? WHERE task_id = ? AND position >= ?",
            (len(parsed), task_id, start),
        )
        ids = _insert(conn, task_id, parsed, start, int(time.time()))
        kb._append_event(conn, task_id, "checklist_added", {
            "count": len(parsed), "position": start, "total": count + len(parsed), "by": _actor(author),
        })
    return ids


def set_items(
    conn: sqlite3.Connection, task_id: str, items: Iterable[Any], *, author: Optional[str] = None,
) -> list[int]:
    """Replace the whole list. An item whose (text, kind) exactly matches a
    previously checked item keeps its done state, so re-specifying a card never
    silently un-does verified work."""
    parsed = normalize_items(items)
    with kb.write_txn(conn, allow_nested=True):
        kb._require_task(conn, task_id)
        previous = {(i.text, i.kind): i for i in list_items(conn, task_id) if i.done}
        conn.execute("DELETE FROM task_checklist_items WHERE task_id = ?", (task_id,))
        ids = _insert(conn, task_id, parsed, 1, int(time.time()))
        kept = 0
        for item_id, key in zip(ids, parsed):
            old = previous.get(key)
            if old is not None:
                kept += 1
                conn.execute(
                    "UPDATE task_checklist_items SET done_at = ?, done_by = ?, evidence = ? WHERE id = ?",
                    (old.done_at, old.done_by, old.evidence, item_id),
                )
        kb._append_event(conn, task_id, "checklist_set", {
            "total": len(parsed), "kept_done": kept, "by": _actor(author),
        })
    return ids


def _resolve(conn: sqlite3.Connection, task_id: str, position: Optional[int], item_id: Optional[int]) -> ChecklistItem:
    if (position is None) == (item_id is None):
        raise ValueError("pass exactly one of position or item_id")
    if item_id is not None:
        row = conn.execute(
            "SELECT * FROM task_checklist_items WHERE id = ? AND task_id = ?", (int(item_id), task_id),
        ).fetchone()
        where = f"item id {item_id}"
    else:
        row = conn.execute(
            "SELECT * FROM task_checklist_items WHERE task_id = ? AND position = ?", (task_id, int(position)),
        ).fetchone()
        where = f"item #{position}"
    if row is None:
        kb._require_task(conn, task_id)
        raise ValueError(f"task {task_id} has no checklist {where}")
    return _item(row)


def _event_item(item: ChecklistItem) -> dict[str, Any]:
    return {"item_id": item.id, "position": item.position, "kind": item.kind, "text": item.text[:120]}


def check_item(
    conn: sqlite3.Connection, task_id: str, *, position: Optional[int] = None, item_id: Optional[int] = None,
    done_by: Optional[str] = None, evidence: Optional[str] = None,
) -> tuple[ChecklistItem, bool]:
    """Mark an item done; ``(item, changed)``. Re-checking a done item is a
    no-op unless new evidence is supplied (then the evidence is updated)."""
    evidence = _evidence(evidence)
    with kb.write_txn(conn, allow_nested=True):
        item = _resolve(conn, task_id, position, item_id)
        if item.done and (evidence is None or evidence == item.evidence):
            return item, False
        now = item.done_at or int(time.time())
        by = _actor(done_by) or item.done_by
        conn.execute(
            "UPDATE task_checklist_items SET done_at = ?, done_by = ?, evidence = ? WHERE id = ?",
            (now, by, evidence if evidence is not None else item.evidence, item.id),
        )
        kb._append_event(conn, task_id, "checklist_checked", {
            **_event_item(item), "by": by, "evidence": (evidence or "")[:200] or None,
        })
        return _resolve(conn, task_id, None, item.id), True


def uncheck_item(
    conn: sqlite3.Connection, task_id: str, *, position: Optional[int] = None, item_id: Optional[int] = None,
    author: Optional[str] = None,
) -> tuple[ChecklistItem, bool]:
    """Clear an item's done state (and its evidence); ``(item, changed)``."""
    with kb.write_txn(conn, allow_nested=True):
        item = _resolve(conn, task_id, position, item_id)
        if not item.done:
            return item, False
        conn.execute(
            "UPDATE task_checklist_items SET done_at = NULL, done_by = NULL, evidence = NULL WHERE id = ?",
            (item.id,),
        )
        kb._append_event(conn, task_id, "checklist_unchecked", {**_event_item(item), "by": _actor(author)})
        return _resolve(conn, task_id, None, item.id), True


def note_unchecked_on_complete(conn: sqlite3.Connection, task_id: str, run_id: Optional[int], now: int) -> int:
    """Completion rule (inside ``complete_task``'s txn): completion is never
    blocked, but unchecked items are recorded as a ``checklist_incomplete``
    event plus a comment so the gap is visible. Returns the unchecked count."""
    remaining = [i for i in list_items(conn, task_id) if not i.done]
    if not remaining:
        return 0
    kb._append_event(conn, task_id, "checklist_incomplete", {
        "remaining": len(remaining), "items": [_event_item(i) for i in remaining[:CHECKLIST_MAX_ITEMS]],
    }, run_id=run_id)
    lines = [f"- #{i.position} ({_KIND_LABELS[i.kind]}) {i.text}" for i in remaining]
    kb._insert_comment(
        conn, task_id, "kanban",
        f"完了時点で未チェックのチェックリスト項目が {len(remaining)} 件あります:\n" + "\n".join(lines), now,
    )
    return len(remaining)


# --- Rendering ---

def render_context_lines(items: list[ChecklistItem]) -> list[str]:
    """Compact worker-context section; [] when the task has no checklist."""
    if not items:
        return []
    done = sum(1 for i in items if i.done)
    lines = [f"## チェックリスト ({done}/{len(items)} 済)"]
    for i in items:
        mark = "x" if i.done else " "
        evidence = f" — 証跡: {i.evidence[:80]}" if i.done and i.evidence else ""
        lines.append(f"- [{mark}] {i.position}. ({_KIND_LABELS[i.kind]}) {i.text}{evidence}")
    lines.append(
        "_各項目は本当に終わった時点ですぐ kanban_check(item=番号, evidence=...) でチェックする。"
        "(機械) 項目はビルド/テスト/デプロイが実際に成功してから証跡付きで。"
        "再開時は最初の未チェック項目から続け、チェック済み項目はやり直さず書き換えない。_"
    )
    lines.append("")
    return lines


def render_human_lines(items: list[ChecklistItem]) -> list[str]:
    """``hermes kanban show`` / ``checklist`` text rows."""
    out = []
    for i in items:
        mark = "x" if i.done else " "
        tail = ""
        if i.done:
            bits = [b for b in (i.done_by and f"by {i.done_by}", i.evidence and f"evidence: {i.evidence}") if b]
            tail = f"  ({', '.join(bits)})" if bits else ""
        out.append(f"  [{mark}] {i.position:>2}. ({_KIND_LABELS[i.kind]}) {i.text}{tail}  [id {i.id}]")
    return out
