"""Dispatch order, effort routing, and picking a dead run back up."""

import json

import pytest

from myriapod.core.scheduler import Swarm
from myriapod.core.task_tree import TaskStatus, TaskTree
from myriapod.testing import Recorder, SimWorker, scripted_planner_factory


# --------------------------------------------------------------------- #
# Critical path
# --------------------------------------------------------------------- #


def test_a_task_with_a_queue_behind_it_goes_out_first():
    """Wall clock is set by the longest chain, so it must start earliest.

    Task 3 has a two-link chain waiting on it; tasks 1 and 2 have nothing.
    In id order — which is what the scheduler dispatched in — the chain head
    goes out third, and on a saturated pool that pushes the longest path of
    the run to the back of the queue.
    """
    tree = TaskTree("goal")
    tree.decompose(
        "root",
        [
            {"description": "independent"},
            {"description": "independent"},
            {"description": "head of a chain"},
            {"description": "link 2", "depends_on": [3]},
            {"description": "link 3", "depends_on": [4]},
        ],
    )
    assert [n.id for n in tree.ready_tasks()] == ["3", "1", "2"]


def test_a_flat_fan_out_keeps_id_order():
    """No dependency anywhere: the ordering must cost nothing and change nothing."""
    tree = TaskTree("goal")
    tree.decompose("root", [{"description": f"t{i}"} for i in range(30)])
    assert [n.id for n in tree.ready_tasks()] == [str(i) for i in range(1, 31)]


def test_a_leaf_inherits_the_queue_waiting_on_its_container():
    """Nothing depending on a section starts before all its leaves are done.

    So a leaf of that section is on the critical path even though nothing
    declares a dependency on the leaf itself — and it must outrank task 1,
    which comes first by id and blocks nobody.
    """
    tree = TaskTree("goal")
    tree.decompose(
        "root",
        [
            {"description": "unrelated"},
            {"description": "section"},
            {"description": "after the section", "depends_on": [2]},
        ],
    )
    tree.decompose("2", [{"description": "leaf"}])
    assert [n.id for n in tree.ready_tasks()] == ["2.1", "1"]


def test_a_cycle_degrades_the_order_rather_than_hanging():
    """decompose rejects cycles; a hand-built or migrated tree might not."""
    tree = TaskTree("goal")
    tree.decompose("root", [{"description": "a"}, {"description": "b"}])
    tree.get("1").depends_on.append("2")
    tree.get("2").depends_on.append("1")
    assert tree.ready_tasks() == []  # blocked, but no crash and no hang


# --------------------------------------------------------------------- #
# Effort routing
# --------------------------------------------------------------------- #


def test_effort_is_recorded_normalized_and_shown_to_the_planner():
    tree = TaskTree("goal")
    tree.decompose(
        "root",
        [
            {"description": "arbitrate", "effort": "high"},
            {"description": "extract", "effort": "LOW"},
            {"description": "ordinary"},
            {"description": "nonsense", "effort": "maximum"},
        ],
    )
    assert [tree.get(i).effort for i in "1234"] == [
        "high", "low", "standard", "standard",
    ]
    rendered = tree.render()
    assert "<high>" in rendered and "<low>" in rendered
    assert "<standard>" not in rendered  # the default costs no tokens


async def test_the_worker_factory_routes_on_the_declared_effort():
    plan = (
        "plan_tasks",
        {
            "parent_id": "root",
            "subtasks": json.dumps(
                [
                    {"description": "cheap", "effort": "low"},
                    {"description": "hard", "effort": "high"},
                ]
            ),
        },
    )
    routed: list[str] = []
    recorder = Recorder()

    def worker_factory(node):
        routed.append(node.effort)
        return SimWorker(node, recorder)

    swarm = Swarm(
        planner_factory=scripted_planner_factory(
            [[plan], [("finish", {"final_answer": "done"})]]
        ),
        worker_factory=worker_factory,
    )
    await swarm.arun("goal")
    assert sorted(routed) == ["high", "low"]


# --------------------------------------------------------------------- #
# Resume
# --------------------------------------------------------------------- #


async def test_resume_keeps_finished_work_and_reruns_what_died_mid_flight():
    tree = TaskTree("goal")
    tree.decompose(
        "root",
        [{"description": "a"}, {"description": "b"}, {"description": "c"}],
    )
    tree.mark_done("1", "summary of a", "FULL a", input_tokens=100, cost=0.5)
    tree.mark_in_progress("2")  # the process died here
    snapshot = json.loads(tree.to_json())

    recorder = Recorder()
    swarm = Swarm(
        planner_factory=scripted_planner_factory(
            [[("finish", {"final_answer": "done"})]]
        ),
        worker_factory=lambda node: SimWorker(node, recorder),
    )
    result = await swarm.arun("goal", resume=snapshot)

    assert set(recorder.started) == {"2", "3"}  # task 1 never re-ran
    assert result.tree["nodes"]["1"]["result_full"] == "FULL a"
    assert result.tasks_done == 3
    # The spend of the first run travels with the tree; it is not re-charged.
    assert result.tree["nodes"]["1"]["cost"] == 0.5


def test_reopen_running_only_touches_leaves_that_were_in_flight():
    tree = TaskTree("goal")
    tree.decompose("root", [{"description": "a"}, {"description": "b"}])
    tree.mark_done("1", "s", "f")
    tree.mark_in_progress("2")
    assert tree.reopen_running() == ["2"]
    assert tree.get("2").status is TaskStatus.PENDING
    assert tree.get("1").status is TaskStatus.DONE
    assert tree.reopen_running() == []


def test_a_reloaded_tree_can_still_take_a_planner_turn():
    """from_dict used to drop the turn counters, so resume died on turn one."""
    tree = TaskTree("goal")
    tree.decompose("root", [{"description": "a"}])
    tree.mark_planner_thinking("planner")
    clone = TaskTree.from_dict(json.loads(tree.to_json()))
    clone.mark_planner_thinking("planner")
    clone.record_planner_turn("planner", 10, 5, 0.01, 1.0)
    assert len(clone.planner_turns) == 2


async def test_resume_of_a_finished_tree_finishes_immediately():
    tree = TaskTree("goal")
    tree.decompose("root", [{"description": "a"}])
    tree.mark_done("1", "s", "FULL a")
    recorder = Recorder()
    swarm = Swarm(
        planner_factory=scripted_planner_factory(
            [[("finish", {"from_task_id": "1"})]]
        ),
        worker_factory=lambda node: SimWorker(node, recorder),
    )
    result = await swarm.arun("goal", resume=tree)
    assert recorder.started == {}
    assert result.content == "FULL a"


async def test_a_failed_attempt_is_billed_for_what_it_burned():
    """A worker can spend its whole output budget and return nothing.

    Booking only successes made those tokens invisible to stats() and so to
    max_cost — a run failing expensively looked free until the invoice.
    """

    class Silent(SimWorker):
        async def run(self, message, context=None):
            outcome = await super().run(message, context)
            outcome.content = ""
            return outcome

    recorder = Recorder()
    swarm = Swarm(
        planner_factory=scripted_planner_factory(
            [
                [("plan_tasks", {"parent_id": "root",
                                 "subtasks": json.dumps([{"description": "a"}])})],
                [("skip_task", {"task_id": "1", "reason": "empty twice"})],
                [("finish", {"final_answer": "done"})],
            ]
        ),
        worker_factory=lambda node: Silent(node, recorder),
        worker_pricing={"input": 1.0, "output": 5.0},
        auto_retry=1,
    )
    result = await swarm.arun("goal")
    # Two attempts at 200/100 tokens, none of which produced an answer.
    assert result.costs.workers_output_tokens == 200
    assert result.costs.workers_cost == pytest.approx(2 * (200 + 500) / 1e6)
    # And the planner is told what the silence cost, which is the whole
    # diagnosis: near zero means the call never ran, thousands means the
    # task is too big for one answer.
    assert "100 output tokens" in result.tree["nodes"]["1"]["error"]
    assert "too big for one answer" in result.tree["nodes"]["1"]["error"]


async def test_an_adapter_that_knows_why_the_answer_is_empty_overrides_the_guess():
    """"Answered badly" and "never got to answer" both arrive as "".

    The core can only read the token count, so it guesses the common cause —
    the task is too big, split it. On a thinking model that guess is exactly
    backwards: a worker whose reasoning ate its whole `max_tokens` never
    reached the brief at all, and splitting it wastes another two attempts.
    Only the adapter holds the model settings that tell the two apart, so
    whatever it reports must reach the planner instead of the guess.
    """

    class CeilingBound(SimWorker):
        async def run(self, message, context=None):
            outcome = await super().run(message, context)
            outcome.content = ""
            outcome.empty_reason = "Worker hit its output ceiling (100 of 100)."
            return outcome

    swarm = Swarm(
        planner_factory=scripted_planner_factory(
            [
                [("plan_tasks", {"parent_id": "root",
                                 "subtasks": json.dumps([{"description": "a"}])})],
                [("skip_task", {"task_id": "1", "reason": "no room to think"})],
                [("finish", {"final_answer": "done"})],
            ]
        ),
        worker_factory=lambda node: CeilingBound(node, Recorder()),
        auto_retry=0,
    )
    result = await swarm.arun("goal")
    error = result.tree["nodes"]["1"]["error"]
    assert "hit its output ceiling" in error
    # The misleading default must be gone, not merely appended to.
    assert "too big for one answer" not in error
