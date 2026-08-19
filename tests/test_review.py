"""The optional quality gate: judge, rewrite, and never make things worse."""

import json

import pytest

from myriapod.core.reviewer import parse_verdict
from myriapod.core.scheduler import Swarm
from myriapod.testing import (
    Recorder,
    SimReviewer,
    SimWorker,
    reset_review_counts,
    scripted_planner_factory,
)

PLAN_2 = (
    "plan_tasks",
    {
        "parent_id": "root",
        "subtasks": json.dumps(
            [{"description": "part A"}, {"description": "part B"}]
        ),
    },
)
TURNS = [[PLAN_2], [("finish", {"final_answer": "done"})]]


@pytest.fixture(autouse=True)
def _clean_counters():
    reset_review_counts()
    yield
    reset_review_counts()


def make_swarm(recorder, reviewer_factory, **kw):
    return Swarm(
        planner_factory=scripted_planner_factory(TURNS),
        worker_factory=lambda node: SimWorker(node, recorder),
        reviewer_factory=reviewer_factory,
        worker_pricing={"input": 1.0, "output": 5.0},
        reviewer_pricing={"input": 0.1, "output": 0.5},
        **kw,
    )


def test_parse_verdict_is_forgiving_and_fails_open():
    assert parse_verdict("VERDICT: PASS")[0] is True
    assert parse_verdict("verdict:  revise\n1. Add figures.")[0] is False
    assert parse_verdict("I think it is fine, honestly")[0] is True  # unreadable
    assert parse_verdict("VERDICT: REVISE")[0] is True  # no reason given


async def test_a_passing_review_costs_a_call_and_changes_nothing():
    recorder = Recorder()
    swarm = make_swarm(recorder, lambda node: SimReviewer(node))
    result = await swarm.arun("goal")
    assert result.tasks_done == 2
    assert "FULL RESULT of 1" in json.dumps(result.tree)
    # Two workers at 200/100 tokens plus two reviews at 50/20.
    assert result.costs.workers_input_tokens == 500


async def test_a_rejected_answer_is_rewritten_and_every_call_is_billed():
    recorder = Recorder()
    swarm = make_swarm(recorder, lambda node: SimReviewer(node, reject={"1"}))
    result = await swarm.arun("goal")
    assert result.tasks_done == 2
    # Task 1: one answer (200), one review (50), one rewrite (200). The last
    # rewrite is deliberately not re-reviewed — with no revision left, the
    # verdict could not change anything and would only cost a call.
    assert result.tree["nodes"]["1"]["input_tokens"] == 450
    assert result.tree["nodes"]["2"]["input_tokens"] == 250
    # And each call is billed at the rate of whoever ran it: three worker
    # calls at 1/5 per million, two reviews at 0.1/0.5. Billing a cheap
    # review at the worker's rate is how a cost cap starts lying.
    workers = 3 * (200 * 1.0 + 100 * 5.0)
    reviews = 2 * (50 * 0.1 + 20 * 0.5)
    assert result.costs.workers_cost == pytest.approx((workers + reviews) / 1e6)


async def test_the_rewrite_sees_the_review_and_its_previous_answer():
    seen: list[str] = []

    class Spy(SimWorker):
        async def run(self, message, context=None):
            seen.append(context or "")
            return await super().run(message, context)

    recorder = Recorder()
    swarm = Swarm(
        planner_factory=scripted_planner_factory(TURNS),
        worker_factory=lambda node: Spy(node, recorder),
        reviewer_factory=lambda node: SimReviewer(node, reject={"1"}),
    )
    await swarm.arun("goal")
    revision = [c for c in seen if "rejected" in c]
    assert len(revision) == 1
    assert "name the actors" in revision[0]  # the reviewer's reason
    assert "FULL RESULT of 1" in revision[0]  # the answer being replaced


async def test_a_reviewer_that_blows_up_accepts_the_answer():
    """A judge is an optimisation; it may never be able to fail a run."""

    class Broken:
        name = "broken-reviewer"

        async def run(self, message, context=None):
            raise RuntimeError("provider down")

    recorder = Recorder()
    swarm = make_swarm(recorder, lambda node: Broken())
    result = await swarm.arun("goal")
    assert result.tasks_done == 2
    assert result.tasks_failed == 0


async def test_max_revisions_zero_disables_the_gate_entirely():
    recorder = Recorder()
    swarm = make_swarm(
        recorder, lambda node: SimReviewer(node, reject={"1", "2"}), max_revisions=0
    )
    result = await swarm.arun("goal")
    assert result.tasks_done == 2
    assert result.tree["nodes"]["1"]["input_tokens"] == 200  # no review call


async def test_a_stronger_tier_is_billed_at_its_own_rate():
    """A fleet that routes "high" to a pricier model must say so in the bill."""
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
    recorder = Recorder()
    swarm = Swarm(
        planner_factory=scripted_planner_factory(
            [[plan], [("finish", {"final_answer": "done"})]]
        ),
        worker_factory=lambda node: SimWorker(node, recorder),
        worker_pricing={
            "low": {"input": 1.0, "output": 5.0},
            "high": {"input": 3.0, "output": 15.0},
        },
    )
    result = await swarm.arun("goal")
    cheap = (200 * 1.0 + 100 * 5.0) / 1e6
    dear = (200 * 3.0 + 100 * 15.0) / 1e6
    assert result.tree["nodes"]["1"]["cost"] == pytest.approx(cheap)
    assert result.tree["nodes"]["2"]["cost"] == pytest.approx(dear)
    assert result.costs.workers_cost == pytest.approx(cheap + dear)


async def test_truncation_is_diagnosed_on_the_answering_call_not_the_total(caplog):
    """The number in that warning is compared against the model's ceiling.

    Reporting the task's cumulative spend instead — attempts plus reviews —
    reads as far above any ceiling and points at the wrong setting.
    """

    class NoSummary(SimWorker):
        async def run(self, message, context=None):
            outcome = await super().run(message, context)
            outcome.content = "an answer cut off before its summ"
            return outcome

    recorder = Recorder()
    swarm = make_swarm(recorder, lambda node: SimReviewer(node, reject={"1"}))
    swarm.worker_factory = lambda node: NoSummary(node, recorder)
    with caplog.at_level("WARNING", logger="myriapod"):
        result = await swarm.arun("goal")

    warnings = [r.getMessage() for r in caplog.records if "no <summary>" in r.getMessage()]
    assert warnings and all("(100 output tokens" in w for w in warnings)
    # while the task is still billed for everything it spent
    assert result.tree["nodes"]["1"]["output_tokens"] == 220
