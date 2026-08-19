"""The status line: what a run looks like when there is no graph.

The point of these is narrow but load-bearing: a run with no visible output
is indistinguishable from a hung one, and the phase a run is in comes from
the tree rather than from the scheduler — so the line must keep moving even
while nothing is being counted.
"""

import json

import pytest

from myriapod.core.scheduler import Swarm
from myriapod.progress import SPINNER, VERBS, arun_with_progress, status_text
from myriapod.testing import Recorder, SimWorker, scripted_planner_factory

TURNS = [
    [("plan_tasks", {"parent_id": "root", "subtasks": json.dumps(
        [{"description": "a"}, {"description": "b"}])})],
    [("finish", {"final_answer": "done"})],
]


def _swarm(**kwargs):
    recorder = Recorder()
    return Swarm(
        planner_factory=scripted_planner_factory(TURNS),
        worker_factory=lambda node: SimWorker(node, recorder, delay=0.05),
        max_concurrency=4,
        **kwargs,
    )


def test_status_line_before_the_tree_exists():
    swarm = _swarm()
    assert swarm.tree is None
    assert "starting up" in status_text(swarm, started=0.0, now=1.0).plain


def test_the_line_moves_even_when_no_counter_does():
    """Spinner and verb are the only thing distinguishing live from hung."""
    swarm = _swarm()
    frames = {status_text(swarm, 0.0, now=t / 4).plain[0] for t in range(len(SPINNER) * 4)}
    assert len(frames) > 1
    assert frames <= set(SPINNER)

    early = status_text(swarm, 0.0, now=1.0).plain
    later = status_text(swarm, 0.0, now=42.0).plain
    assert any(v in early for v in VERBS) and any(v in later for v in VERBS)
    assert early.split("…")[0] != later.split("…")[0]


async def test_the_phase_comes_from_the_tree():
    swarm = _swarm()
    result = await arun_with_progress(swarm, "goal", quiet=True)
    assert result.finished

    line = status_text(swarm, 0.0, now=1.0).plain
    assert "2 tasks" in line and "✔    2" in line

    # A planner turn in flight is the one phase no counter reveals: the
    # tree flags it at the start of the turn, tokens only arrive at the end.
    swarm.tree.mark_planner_thinking("planner-model")
    assert "planning" in status_text(swarm, 0.0, now=1.0).plain


async def test_quiet_and_non_interactive_runs_return_the_same_result():
    swarm = _swarm()
    result = await arun_with_progress(swarm, "goal", quiet=True)
    assert result.finished and result.tasks_done == 2
    assert result.content == "done"


async def test_progress_does_not_swallow_a_failing_run():
    """A crash inside the run must surface, not be eaten by the display."""

    def exploding_factory(tools):
        raise RuntimeError("no planner today")

    swarm = Swarm(
        planner_factory=exploding_factory,
        worker_factory=lambda node: SimWorker(node, Recorder()),
    )
    with pytest.raises(RuntimeError, match="no planner today"):
        await arun_with_progress(swarm, "goal", quiet=True)
