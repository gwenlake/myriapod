"""Watch a swarm think — 60 agents in the browser, for free.

Same doubles as `01_offline_swarm.py`, but wired to the live graph and shaped
like a real report: three planner waves, four synthesis tasks pulling on the
leaves. Nothing here calls a model, so you can leave it running, resize,
click nodes and try the layouts without spending anything.

    uv run python examples/02_live_graph.py

The page opens on http://127.0.0.1:8400. Each node is one agent: the border
is its status, the fill is its dependency group, and the run reads left to
right — one column of workers per planner turn. Click a node for its tokens,
cost, duration and summary.
"""

import asyncio

from myriapod.core.scheduler import Swarm
from myriapod.testing import Recorder, SimWorker, fanout_turns, scripted_planner_factory
from myriapod.viz import serve

TASKS, WAVES, GROUPS = 60, 3, 4


async def main() -> None:
    recorder = Recorder()
    swarm = Swarm(
        planner_factory=scripted_planner_factory(
            fanout_turns(TASKS, groups=GROUPS, waves=WAVES),
            delay=2.0,  # the root goes amber while the turn is "thinking"
        ),
        worker_factory=lambda node: SimWorker(node, recorder, delay=1.5, jitter=1.0),
        max_concurrency=10,  # low on purpose: the frontier stays visible
        max_tasks=TASKS + GROUPS + 10,
        max_planner_turns=WAVES + 3,
        worker_pricing={"input": 1.0, "output": 5.0},
    )

    with serve(swarm) as url:
        print(f"live graph: {url}")
        result = await swarm.arun("Simulated run: a report in three waves")
        print(f"{result.reason} — {result.tasks_done} tasks, "
              f"max {recorder.max_parallel} workers at once")
        input("\nthe page stays live — press Enter to quit ")


if __name__ == "__main__":
    asyncio.run(main())
