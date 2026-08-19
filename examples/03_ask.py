"""A real run, kept small on purpose — a few cents, a couple of minutes.

Sonnet plans, a Haiku fleet executes. The goal is deliberately narrow: five
or six briefs is enough to see the shape of a run without paying for one.
`max_cost` is the hard ceiling — the run stops and returns what it has
rather than overrunning.

    export ANTHROPIC_API_KEY=...
    uv run python examples/03_ask.py

The same run from the shell, with the defaults:

    uv run myriapod ask "..." --max-cost 0.4
"""

from myriapod.adapters.pydantic_ai import build_swarm
from myriapod.progress import run_with_progress

swarm = build_swarm(
    # The planner never sees a worker's raw output — only statuses and
    # summaries — so it stays cheap even when the fleet is wide.
    planner_model="anthropic:claude-sonnet-5",
    worker_model="anthropic:claude-haiku-4-5",
    max_concurrency=8,        # simultaneous API calls; raise on a paid tier
    max_cost=0.40,            # hard $ ceiling, enforced as the run goes
    max_planner_turns=6,
    worker_timeout=180,
    # $ per million tokens. Without pricing the run works but bills $0.00,
    # and max_cost cannot bite — the library never guesses a price.
    planner_pricing={"input": 3.0, "output": 15.0},
    worker_pricing={"input": 1.0, "output": 5.0},
)

# `run_with_progress` is `swarm.run` plus a live status line on stderr.
result = run_with_progress(
    swarm,
    "Compare SQLite, DuckDB and Postgres for a single-node analytics service "
    "handling 50 GB of event data: storage model, concurrency, query "
    "performance, operational cost. End with a recommendation."
)

print(result.content)
print(
    f"\n{result.reason} — {result.tasks_done} tasks, {result.planner_turns} planner "
    f"turns, {result.duration:.0f}s, ${result.costs.total_cost:.4f} "
    f"(planner {result.costs.planner_share:.0%})"
)
