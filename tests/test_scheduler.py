import asyncio
import json

import pytest

from myriapod.core.scheduler import Swarm
from myriapod.testing import (
    Recorder,
    ScriptedPlanner,
    SimWorker,
    fanout_turns,
    scripted_planner_factory,
)

pytestmark = pytest.mark.asyncio

PLAN_3 = (
    "plan_tasks",
    {
        "parent_id": "root",
        "subtasks": json.dumps(
            [
                {"description": "research part A"},
                {"description": "research part B"},
                {"description": "final synthesis", "depends_on": [1, 2]},
            ]
        ),
    },
)


def make_swarm(turns, recorder, **worker_kw):
    extra = worker_kw.pop("swarm_kw", {})
    return Swarm(
        planner_factory=scripted_planner_factory(turns),
        worker_factory=lambda n: SimWorker(n, recorder, **worker_kw),
        planner_pricing={"input": 10.0, "output": 30.0},
        worker_pricing={"input": 1.0, "output": 5.0},
        **extra,
    )


async def test_happy_path_parallelism_costs_and_isolation():
    recorder = Recorder()
    turns = [[PLAN_3], [("finish", {"from_task_id": "3"})]]
    swarm = make_swarm(turns, recorder, swarm_kw={"max_concurrency": 4})
    result = await swarm.arun("Write a two-part report")

    assert result.finished and result.reason == "finished"
    assert result.tasks_done == 3 and result.tasks_failed == 0
    assert result.planner_turns == 2
    assert result.content.startswith("FULL RESULT of 3")
    assert recorder.max_parallel >= 2
    assert recorder.started["3"] >= max(recorder.ended["1"], recorder.ended["2"])
    assert result.costs.planner_input_tokens == 2000
    assert result.costs.workers_input_tokens == 600
    assert result.costs.planner_cost == pytest.approx(2 * (1000 * 10 + 200 * 30) / 1e6)
    assert result.costs.workers_cost == pytest.approx(3 * (200 * 1 + 100 * 5) / 1e6)
    # Per-task cost recorded in the tree.
    assert result.tree["nodes"]["1"]["cost"] == pytest.approx((200 * 1 + 100 * 5) / 1e6)


async def test_planner_never_sees_raw_outputs():
    recorder = Recorder()
    turns = [[PLAN_3], [("finish", {"from_task_id": "3"})]]
    planner_holder = {}

    def factory(tools):
        planner = ScriptedPlanner(list(turns))
        planner.tools = tools
        planner_holder["p"] = planner
        return planner

    swarm = Swarm(
        planner_factory=factory,
        worker_factory=lambda n: SimWorker(n, recorder),
    )
    result = await swarm.arun("goal")
    assert result.finished
    assert not any("FULL RESULT" in m for m in planner_holder["p"].messages)


async def test_auto_retry_recovers_transient_failure():
    recorder = Recorder()
    turns = [[PLAN_3], [("finish", {"from_task_id": "3"})]]
    swarm = make_swarm(turns, recorder, fail_first={"2"}, swarm_kw={"auto_retry": 1})
    result = await swarm.arun("goal")
    assert result.finished
    assert result.tasks_failed == 0
    assert result.tree["nodes"]["2"]["attempts"] == 2
    assert result.planner_turns == 2


async def test_budget_stops_the_run():
    recorder = Recorder()
    swarm = make_swarm([[PLAN_3]], recorder, swarm_kw={"max_cost": 1e-9})
    result = await swarm.arun("goal")
    assert not result.finished
    assert result.reason == "max_cost"


async def test_planner_stall_is_detected():
    recorder = Recorder()
    swarm = make_swarm([[], [], []], recorder)
    result = await swarm.arun("goal")
    assert not result.finished
    assert result.reason == "planner_stalled"
    assert result.planner_turns == 2


async def test_worker_timeout_then_planner_repairs():
    recorder = Recorder()
    turns = [
        [PLAN_3],
        [("skip_task", {"task_id": "1", "reason": "kept timing out"})],
        [("finish", {"final_answer": "partial report without part A"})],
    ]
    swarm = make_swarm(
        turns,
        recorder,
        sleep_forever={"1"},
        swarm_kw={"worker_timeout": 0.2, "auto_retry": 0},
    )
    result = await swarm.arun("goal")
    assert result.finished
    assert result.tree["nodes"]["1"]["status"] == "skipped"
    assert result.content == "partial report without part A"


async def test_max_tasks_guard():
    recorder = Recorder()
    swarm = Swarm(
        planner_factory=scripted_planner_factory(fanout_turns(30, batch=10)),
        worker_factory=lambda n: SimWorker(n, recorder, delay=0.001),
        max_tasks=20,
    )
    result = await swarm.arun("goal")
    assert not result.finished
    assert result.reason == "max_tasks"


async def test_fanout_500_tasks_high_concurrency():
    """The 'hundreds of agents' guarantee, in-suite and fast."""
    recorder = Recorder()
    swarm = Swarm(
        planner_factory=scripted_planner_factory(fanout_turns(500, batch=50)),
        worker_factory=lambda n: SimWorker(n, recorder, delay=0.01, jitter=0.01),
        max_concurrency=250,
        max_planner_turns=4,
    )
    result = await swarm.arun("bench")
    assert result.finished
    assert result.tasks_done == 500
    assert recorder.max_parallel > 100  # real parallelism, not sequential
    assert result.duration < 10


async def test_sync_run_wrapper():
    recorder = Recorder()
    turns = [[PLAN_3], [("finish", {"from_task_id": "3"})]]
    swarm = make_swarm(turns, recorder)
    result = await asyncio.to_thread(swarm.run, "goal")
    assert result.finished


class ExplodingPlanner(ScriptedPlanner):
    """Planner double whose first ``n_failures`` turns raise."""

    def __init__(self, turns, n_failures: int):
        super().__init__(turns)
        self.n_failures = n_failures
        self.calls = 0

    async def run(self, message: str, context: str | None = None):
        self.calls += 1
        if self.calls <= self.n_failures:
            raise RuntimeError("Tool 'plan_tasks' exceeded max retries count of 1")
        return await super().run(message, context)


def exploding_factory(turns, n_failures: int):
    def factory(tools: list) -> ExplodingPlanner:
        planner = ExplodingPlanner(turns, n_failures)
        planner.tools = tools
        return planner

    return factory


async def test_planner_exception_is_retried_not_fatal():
    recorder = Recorder()
    turns = [[PLAN_3], [("finish", {"from_task_id": "3"})]]
    swarm = Swarm(
        planner_factory=exploding_factory(turns, n_failures=1),
        worker_factory=lambda n: SimWorker(n, recorder, delay=0.001),
    )
    result = await swarm.arun("goal")
    assert result.finished, result.reason
    assert result.tasks_done == 3
    assert any(entry["action"] == "planner_error" for entry in result.log)


async def test_persistent_planner_exception_ends_the_run_cleanly():
    recorder = Recorder()
    swarm = Swarm(
        planner_factory=exploding_factory([[PLAN_3]], n_failures=99),
        worker_factory=lambda n: SimWorker(n, recorder, delay=0.001),
        max_planner_errors=2,
    )
    result = await swarm.arun("goal")   # must not raise
    assert result.reason == "planner_error"
    assert not result.finished
    assert result.content is None


class NoSummaryWorker(SimWorker):
    """Worker double whose answer is cut off before the summary block."""

    async def run(self, message: str, context: str | None = None):
        outcome = await super().run(message, context)
        outcome.content = "Un début de rapport qui s'arrête au milieu d'une phr"
        return outcome


async def test_missing_summary_block_is_flagged_to_the_planner():
    from myriapod.core.worker import TRUNCATED_MARKER

    recorder = Recorder()
    turns = [
        [("plan_tasks", {"parent_id": "root",
                         "subtasks": json.dumps([{"description": "write it all"}])})],
        [("finish", {"from_task_id": "1"})],
    ]
    swarm = Swarm(
        planner_factory=scripted_planner_factory(turns),
        worker_factory=lambda n: NoSummaryWorker(n, recorder, delay=0.001),
    )
    result = await swarm.arun("goal")
    node = result.tree["nodes"]["1"]
    assert node["status"] == "done"
    assert node["result_summary"].startswith(TRUNCATED_MARKER)
    # the marker has to survive into what the planner actually reads
    assert TRUNCATED_MARKER in swarm.tree.render()


async def test_finish_in_the_planning_turn_is_refused():
    """A planner that plans and finishes in one turn would end the run before
    any worker ran: the plan never executes and the "answer" is the planner's
    own guess. finish() must refuse while the frontier is not drained."""
    recorder = Recorder()
    turns = [
        [PLAN_3, ("finish", {"final_answer": "premature guess"})],
        [("finish", {"from_task_id": "3"})],
    ]
    swarm = make_swarm(turns, recorder, swarm_kw={"max_concurrency": 4})
    result = await swarm.arun("Write a two-part report")

    assert result.finished and result.reason == "finished"
    assert result.tasks_done == 3            # the plan actually executed
    assert result.planner_turns == 2         # and the planner was called back
    assert "premature guess" not in result.content
    assert result.content.startswith("FULL RESULT of 3")


async def test_finish_without_tasks_still_works():
    """A trivial goal the planner answers itself must still finish on turn 1."""
    recorder = Recorder()
    swarm = make_swarm([[("finish", {"final_answer": "42"})]], recorder)
    result = await swarm.arun("What is six times seven?")

    assert result.finished and result.planner_turns == 1
    assert result.content == "42"


async def test_workers_start_while_the_planner_turn_is_still_running():
    """A planner writing a 50-task plan takes minutes. Its tasks land in the
    tree as it calls plan_tasks, so the fleet must start on them instead of
    waiting for the turn to return — the difference between working and
    idling for the length of the plan."""
    recorder = Recorder()
    factory = scripted_planner_factory(
        [[PLAN_3], [("finish", {"from_task_id": "3"})]], delay=0.4
    )
    swarm = Swarm(
        planner_factory=factory,
        worker_factory=lambda n: SimWorker(n, recorder, delay=0.02),
        max_concurrency=4,
    )
    result = await swarm.arun("Write a two-part report")

    assert result.finished and result.tasks_done == 3
    planner = factory.planners[0]
    first_turn_returned = planner.returned_at[0]
    # tasks 1 and 2 are ready the moment plan_tasks lands
    assert recorder.started["1"] < first_turn_returned
    assert recorder.ended["1"] < first_turn_returned


async def test_root_carries_the_planner_identity_and_spend():
    """The planner has no leaf of its own, so the root is where it shows up:
    running (and named) while a turn is in flight, with its tokens and cost
    once the turn returns. Before this, every planner figure read zero."""
    recorder = Recorder()
    factory = scripted_planner_factory(
        [[PLAN_3], [("finish", {"from_task_id": "3"})]], delay=0.3
    )
    swarm = Swarm(
        planner_factory=factory,
        worker_factory=lambda n: SimWorker(n, recorder, delay=0.02),
        max_concurrency=4,
        planner_pricing={"input": 10.0, "output": 30.0},
        worker_pricing={"input": 1.0, "output": 5.0},
    )
    seen = {}

    async def watch():
        # while the first turn is still thinking
        await asyncio.sleep(0.1)
        root = swarm.tree.get("root")
        seen["status"] = root.status.value
        seen["worker"] = root.worker

    await asyncio.gather(swarm.arun("Write a two-part report"), watch())

    assert seen["status"] == "in_progress"      # visibly working, not idle
    assert "scripted-planner" in seen["worker"]  # named before the turn ends

    root = swarm.tree.get("root")
    assert root.attempts == 2                    # one entry per turn
    assert root.input_tokens == 2000 and root.output_tokens == 400
    assert root.cost > 0 and root.duration > 0
    # and the tree's headline cost includes it, without double counting
    stats = swarm.tree.stats()
    assert stats["cost"] == pytest.approx(stats["workers_cost"] + root.cost)
    assert len(swarm.tree.planner_turns) == 2
    assert swarm.tree.planner_turns[0]["tasks"] == 3


async def test_successive_planning_waves_run_one_after_another():
    """The end-to-end shape of iterative planning: plan a batch, let it run,
    plan the next from what came back. bench --waves drives exactly this."""
    recorder = Recorder()
    swarm = make_swarm(
        fanout_turns(9, groups=2, waves=3), recorder, swarm_kw={"max_concurrency": 4}
    )
    result = await swarm.arun("bench")

    assert result.finished
    assert result.tasks_done == 11               # 9 leaves + 2 synthesis tasks
    assert result.planner_turns == 4             # three waves, then finish
    planned = [t["tasks"] for t in swarm.tree.planner_turns]
    assert planned == [3, 3, 5, 0]               # last wave carries the synthesis
