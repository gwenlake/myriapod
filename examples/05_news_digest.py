"""A news digest: one search agent per angle, one editor to reconcile them.

The shape worth copying is the fan-out. Rather than asking one agent to
"summarise the news", the planner is told to open one task per *angle* —
each worker searches on its own, and a final task reads only their summaries
and resolves the contradictions. Workers never see each other's raw output
unless they declared the dependency, so ten searches cost ten small contexts,
not one enormous one.

Needs the `web` extra for DuckDuckGo search:

    uv sync --extra web
    export ANTHROPIC_API_KEY=...
    uv run python examples/05_news_digest.py "European AI regulation"
"""

import sys

from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

from myriapod.adapters.pydantic_ai import build_swarm
from myriapod.progress import run_with_progress

TOPIC = sys.argv[1] if len(sys.argv) > 1 else "European AI regulation"

GOAL = f"""Produce a news digest on: {TOPIC}.

Plan one task per distinct angle — regulation, funding and deals, product
launches, notable criticism, and anything the searches surface that these
miss. Each task searches the web itself and reports what it found with dates
and sources. Then plan a final task, depending on all of them, that writes
the digest: what happened, what it means, and where the sources disagree.

Every claim carries a date and a source URL. A worker that finds nothing says
so — an empty angle is a finding, an invented one is a defect."""


def main() -> None:
    swarm = build_swarm(
        planner_model="anthropic:claude-sonnet-5",
        worker_model="anthropic:claude-haiku-4-5",
        # Search is what workers do here, so give them the tool. The planner
        # gets none: it plans on summaries and never gathers anything itself.
        worker_tools=[duckduckgo_search_tool()],
        max_concurrency=6,
        max_cost=0.50,
        worker_timeout=240,          # a search round-trip is slower than a call
        planner_pricing={"input": 3.0, "output": 15.0},
        worker_pricing={"input": 1.0, "output": 5.0},
    )

    result = run_with_progress(swarm, GOAL)

    print(result.content or f"no answer: {result.reason}")
    print(f"\n{result.reason} — {result.tasks_done} tasks, "
          f"${result.costs.total_cost:.4f}")


if __name__ == "__main__":
    main()
