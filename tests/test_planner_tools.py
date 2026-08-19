"""The planner's four tools, called directly — no model, no framework.

These pin the forgiving-argument contract: a model that hits its output
ceiling mid-call must get back something it can act on, not a framework
TypeError it will answer by retrying the same oversized call.
"""

import json

import pytest

from myriapod.core.planner import SwarmState, make_planner_tools
from myriapod.core.task_tree import TaskTree
from myriapod.core.worker import TRUNCATED_MARKER


@pytest.fixture
def tools():
    tree = TaskTree("goal")
    state = SwarmState()
    plan_tasks, retry_task, skip_task, finish = make_planner_tools(tree, state)
    return tree, state, plan_tasks


def test_plan_tasks_accepts_a_list_a_json_string_and_a_lone_object(tools):
    tree, _, plan_tasks = tools
    plan_tasks("root", [{"description": "a"}])
    plan_tasks("root", json.dumps([{"description": "b"}]))
    plan_tasks("root", {"description": "c"})
    assert [tree.get(i).description for i in ("1", "2", "3")] == ["a", "b", "c"]


def test_batches_append_and_can_depend_on_earlier_ids(tools):
    tree, _, plan_tasks = tools
    plan_tasks("root", [{"description": "a"}, {"description": "b"}])
    plan_tasks("root", [{"description": "synthesis", "depends_on": ["1", "2"]}])
    assert tree.get("3").depends_on == ["1", "2"]


@pytest.mark.parametrize("subtasks", [None, "", "   "])
def test_a_truncated_call_gets_a_batching_hint_not_a_crash(tools, subtasks):
    """Anthropic drops a tool argument left unfinished by the output limit.

    The call then arrives as plan_tasks(parent_id="root") alone. Signature
    defaults keep that from raising inside the framework, and the answer
    tells the planner why it lost the batch and what to send instead.
    """
    tree, state, plan_tasks = tools
    answer = plan_tasks("root", subtasks)
    assert "cut off" in answer and "AT MOST" in answer
    assert len(tree) == 0
    assert state.planned_something is False


def test_unterminated_json_is_reported_as_truncation(tools):
    _, _, plan_tasks = tools
    answer = plan_tasks("root", '[{"description": "a long brief that stops mi')
    assert "cut off" in answer


def test_malformed_but_complete_json_is_reported_as_a_typo(tools):
    """A closed fragment is a mistake to fix, not a batch to split."""
    _, _, plan_tasks = tools
    answer = plan_tasks("root", '[{"description": }]')
    assert "cut off" not in answer
    assert "Fix the JSON" in answer


def test_finish_pushes_back_once_on_a_truncated_deliverable():
    """The marker exists so a cut-off answer never ships looking finished.

    finish(from_task_id=...) was the one door that never looked at it — and
    it is the door the deliverable goes through.
    """
    tree = TaskTree("goal")
    state = SwarmState()
    _, _, _, finish = make_planner_tools(tree, state)
    tree.decompose("root", [{"description": "the report"}])
    tree.mark_done("1", f"{TRUNCATED_MARKER} a report that stops mid-", "## Part one")

    first = finish(from_task_id="1")
    assert "cut off" in first and "same from_task_id again" in first
    assert state.finished is False
    assert state.final_answer is None

    # Asking again is a decision, not an oversight: a turn-starved run must
    # still be able to ship the long cut-off answer rather than summaries.
    second = finish(from_task_id="1")
    assert state.finished is True
    assert state.final_answer == "## Part one"
    assert {"action": "finish_truncated", "task": "1"} in state.log
    assert "completed" in second


def test_finish_on_an_intact_task_is_untouched():
    tree = TaskTree("goal")
    state = SwarmState()
    _, _, _, finish = make_planner_tools(tree, state)
    tree.decompose("root", [{"description": "the report"}])
    tree.mark_done("1", "a complete digest", "## The whole report")
    assert "completed" in finish(from_task_id="1")
    assert state.final_answer == "## The whole report"
