"""The whole machine, with no model in it — free, offline, no API key.

`myriapod.testing` ships the two doubles the test suite runs on: a planner
that executes a scripted list of tool calls, and a worker that sleeps and
answers. Swapping them in for real agents changes nothing else, so this is
the shortest way to see how a run is actually shaped: a planner turn writes
tasks, workers drain them in parallel, a synthesis task waits on its
dependencies, and a last planner turn calls `finish`.

    uv run python examples/01_offline_swarm.py
"""

import asyncio
import json

from myriapod.core.scheduler import Swarm
from myriapod.progress import arun_with_progress
from myriapod.testing import Recorder, SimWorker, scripted_planner_factory

# One planner turn per list. Each entry is a tool call the planner makes —
# the same four tools a real planner drives the tree with.
TURNS = [
    # Turn 1: four independent briefs, then one that waits on all of them.
    [
        ("plan_tasks", {
            "parent_id": "root",
            "subtasks": json.dumps([
                {"description": "Survey the open-source landscape"},
                {"description": "Profile the three main vendors"},
                {"description": "Collect 2026 pricing"},
                {"description": "List the integration constraints"},
            ]),
        }),
        ("plan_tasks", {
            "parent_id": "root",
            # A later batch may depend on ids an earlier one returned.
            "subtasks": json.dumps([
                {
                    "description": "Synthesise a recommendation",
                    "depends_on": ["1", "2", "3", "4"],
                    "effort": "high",
                },
            ]),
        }),
    ],
    # Turn 2: the frontier is empty, the planner is woken and ends the run.
    [("finish", {"final_answer": "Recommendation: adopt B, pilot C."})],
]


async def main() -> None:
    recorder = Recorder()  # instrumentation shared by the simulated workers
    swarm = Swarm(
        planner_factory=scripted_planner_factory(TURNS),
        worker_factory=lambda node: SimWorker(node, recorder, delay=0.4),
        max_concurrency=8,
        # Simulated tokens, billed at a made-up rate so the numbers move.
        planner_pricing={"input": 5.0, "output": 25.0},
        worker_pricing={"input": 1.0, "output": 5.0},
    )

    # `arun_with_progress` is `swarm.arun` with a live status line: without
    # the graph, a terminal that prints nothing looks like one that hung.
    result = await arun_with_progress(
        swarm, "Which agent framework should we standardise on?"
    )

    print(result.content)
    print(
        f"\n{result.reason} — {result.tasks_done} tasks in {result.duration:.1f}s, "
        f"{result.planner_turns} planner turns, "
        f"max {recorder.max_parallel} workers at once, "
        f"${result.costs.total_cost:.4f}"
    )

    # The tree is the run's memory: statuses, summaries and full outputs.
    # `render()` is the *only* thing the planner ever sees of it.
    print("\nWhat the planner sees:\n")
    print(swarm.tree.render())


if __name__ == "__main__":
    asyncio.run(main())
