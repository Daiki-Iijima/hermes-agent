"""Transcript evidence must identify a successful lifecycle call for this worker."""

import json
from types import SimpleNamespace

import pytest

from agent.kanban_stop import (
    build_kanban_stop_nudge,
    kanban_stop_nudge_enabled,
    session_called_kanban_terminal,
)


@pytest.fixture
def worker_identity(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "guard-test")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "17")
    monkeypatch.delenv("HERMES_KANBAN_STOP_NUDGE", raising=False)


@pytest.mark.parametrize("disabled", ["0", "false", "no", "off"])
def test_env_can_disable(worker_identity, monkeypatch, disabled):
    monkeypatch.setenv("HERMES_KANBAN_STOP_NUDGE", disabled)
    assert kanban_stop_nudge_enabled() is False
    assert build_kanban_stop_nudge(messages=[]) is None


@pytest.mark.parametrize("name", [
    "kanban_complete", "kanban_block", "kanban_request_review", "kanban_request_changes",
])
@pytest.mark.parametrize("object_call", [False, True])
def test_success_requires_matching_call_result(worker_identity, name, object_call):
    args = {"task_id": "t_worker", "board": "guard-test"}
    call = {"id": "call-1", "function": {"name": name, "arguments": json.dumps(args)}}
    if object_call:
        call = SimpleNamespace(id=call["id"], function=SimpleNamespace(**call["function"]))
    messages = [{"role": "assistant", "tool_calls": [call]}]
    result = {"role": "tool", "tool_call_id": "call-1", "content": json.dumps({
        "ok": True, "task_id": "t_worker", "run_id": 17,
    })}
    assert session_called_kanban_terminal(messages) is False
    assert session_called_kanban_terminal([result]) is False
    assert session_called_kanban_terminal(messages + [result]) is True


@pytest.mark.parametrize("case", [
    "failed", "error", "text", "malformed", "array", "wrong_call", "wrong_name",
    "wrong_task_arg", "wrong_board_arg", "wrong_task_result", "wrong_run_result",
    "wrong_board_result", "no_task_result", "bad_args", "heartbeat",
])
def test_unrelated_or_failed_results_are_not_success(worker_identity, case):
    args = {}
    payload = {"ok": True, "task_id": "t_worker", "run_id": 17}
    result = {"role": "tool", "tool_call_id": "call-1"}
    if case == "failed":
        payload["ok"] = False
    if case == "error":
        payload["error"] = "rejected"
    if case == "wrong_task_arg":
        args["task_id"] = "t_other"
    if case == "wrong_board_arg":
        args["board"] = "other"
    if case == "wrong_task_result":
        payload["task_id"] = "t_other"
    if case == "wrong_run_result":
        payload["run_id"] = 18
    if case == "wrong_board_result":
        payload["board"] = "other"
    if case == "no_task_result":
        payload.pop("task_id")
    if case == "wrong_call":
        result["tool_call_id"] = "unrelated"
    if case == "wrong_name":
        result["name"] = "kanban_block"
    result["content"] = {
        "text": "done", "malformed": "{", "array": "[]",
    }.get(case, json.dumps(payload))
    messages = [{"role": "assistant", "tool_calls": [{
        "id": "call-1", "function": {
            "name": "kanban_heartbeat" if case == "heartbeat" else "kanban_complete",
            "arguments": "{" if case == "bad_args" else json.dumps(args),
        },
    }]}, result]
    assert session_called_kanban_terminal(messages) is False
