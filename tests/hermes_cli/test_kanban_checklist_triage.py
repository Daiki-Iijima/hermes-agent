"""Decomposer / specifier checklist output: parsed and stored; missing or malformed never fails triage."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_checklist as kbcl
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_decompose as decomp
from hermes_cli import kanban_specify as spec


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _llm(payload) -> patch:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return patch("agent.auxiliary_client.call_llm", return_value=resp)


@pytest.fixture
def profiles():
    names = ["orchestrator", "researcher", "engineer"]
    fake = [SimpleNamespace(name=n, description=f"desc for {n}") for n in names]
    with patch("hermes_cli.profiles.list_profiles", return_value=fake), \
            patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x in names), \
            patch("hermes_cli.profiles.get_active_profile_name", return_value="orchestrator"):
        yield


def _triage(title="ship it") -> str:
    with kbc.connect_closing() as conn:
        return kb.create_task(conn, title=title, triage=True)


def _items(tid: str) -> list[tuple[str, str]]:
    with kbc.connect_closing() as conn:
        return [(i.kind, i.text) for i in kbcl.list_items(conn, tid)]


def test_prompts_ask_for_checklists():
    assert '"checklist"' in decomp._SYSTEM_PROMPT and "3-8" in decomp._SYSTEM_PROMPT
    assert '"checklist"' in spec._SYSTEM_PROMPT and "machine" in spec._SYSTEM_PROMPT


def test_decompose_stores_valid_and_skips_malformed_checklists(kanban_home, profiles, caplog):
    tid = _triage()
    payload = {"fanout": True, "rationale": "r", "tasks": [
        {"title": "調査", "body": "b", "assignee": "researcher", "parents": [],
         "checklist": [{"text": "既存実装を読む", "kind": "ai"}, {"text": "pytest を実行", "kind": "machine"}]},
        {"title": "実装", "body": "b", "assignee": "engineer", "parents": [0],
         "checklist": [{"text": "ok step"}, {"text": "", "kind": "ai"}, {"text": "x", "kind": "robot"}, 42]},
        {"title": "no list", "body": "b", "assignee": "engineer", "parents": []},
        {"title": "bad list", "body": "b", "assignee": "engineer", "parents": [], "checklist": "run tests"},
    ]}
    with _llm(payload), caplog.at_level(logging.WARNING, logger="hermes_cli.kanban_db_checklist"):
        outcome = decomp.decompose_task(tid, author="me")
    assert outcome.ok, outcome.reason
    c0, c1, c2, c3 = outcome.child_ids
    assert _items(c0) == [("ai", "既存実装を読む"), ("machine", "pytest を実行")]
    assert _items(c1) == [("ai", "ok step")]
    assert _items(c2) == [] and _items(c3) == []
    assert "skipping checklist[1]" in caplog.text and "checklist is not a list" in caplog.text
    with kbc.connect_closing() as conn:
        assert "## チェックリスト (0/2 済)" in kb.build_worker_context(conn, c0)


def test_decompose_caps_checklist_at_limit(kanban_home, profiles):
    tid = _triage()
    payload = {"fanout": True, "tasks": [{"title": "t", "assignee": "engineer", "parents": [],
                                          "checklist": [{"text": f"s{n}", "kind": "ai"} for n in range(40)]}]}
    with _llm(payload):
        outcome = decomp.decompose_task(tid, author="me")
    assert outcome.ok and len(_items(outcome.child_ids[0])) == kbcl.CHECKLIST_MAX_ITEMS


def test_decompose_fanout_false_stores_checklist(kanban_home, profiles):
    tid = _triage()
    payload = {"fanout": False, "title": "Tight", "body": "spec", "assignee": "engineer",
               "checklist": [{"text": "implement", "kind": "ai"}, {"text": "type-check", "kind": "machine"}]}
    with _llm(payload):
        outcome = decomp.decompose_task(tid, author="me")
    assert outcome.ok and not outcome.fanout
    assert _items(tid) == [("ai", "implement"), ("machine", "type-check")]


def test_specify_stores_checklist_and_tolerates_missing_or_malformed(kanban_home):
    good, missing, broken = _triage("a"), _triage("b"), _triage("c")
    with _llm({"title": "A", "body": "spec", "checklist": [{"text": "build", "kind": "machine"}]}):
        assert spec.specify_task(good, author="me").ok
    with _llm({"title": "B", "body": "spec"}):
        assert spec.specify_task(missing, author="me").ok
    with _llm({"title": "C", "body": "spec", "checklist": {"text": "nope"}}):
        assert spec.specify_task(broken, author="me").ok
    assert _items(good) == [("machine", "build")]
    assert _items(missing) == [] and _items(broken) == []
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, broken).status in {"todo", "ready"}


def test_specify_never_replaces_an_existing_checklist(kanban_home):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="has list", triage=True, checklist=["ai:human step"])
    with _llm({"title": "T", "body": "spec", "checklist": [{"text": "llm step", "kind": "ai"}]}):
        assert spec.specify_task(tid, author="me").ok
    assert _items(tid) == [("ai", "human step")]
