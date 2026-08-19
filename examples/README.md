# Examples

Six short programs, in the order they are worth reading. The first two cost
nothing — they run the whole machine on simulated agents, so no API key is
involved. The rest make real calls, each under a hard `max_cost` set in the
file itself.

| | What it shows | Cost |
| --- | --- | --- |
| [`01_offline_swarm.py`](01_offline_swarm.py) | The shape of a run: a planner turn writes tasks, workers drain them in parallel, one task waits on its dependencies, a last turn calls `finish`. Ends by printing what the planner actually sees of the tree. | free |
| [`02_live_graph.py`](02_live_graph.py) | The same machine, 60 agents, three planner waves, wired to the live browser graph. | free |
| [`03_ask.py`](03_ask.py) | A real run: Sonnet plans, a Haiku fleet executes, `max_cost` caps it. | ≤ $0.40 |
| [`04_report_and_resume.py`](04_report_and_resume.py) | Rendering the answer to Markdown and PDF, and resuming an interrupted run from its saved tree. | ≤ $0.60 |
| [`05_news_digest.py`](05_news_digest.py) | Web search in the fleet: one searching worker per angle, one editor reconciling them. Needs the `web` extra. | ≤ $0.50 |
| [`06_market_brief.py`](06_market_brief.py) | A worker tool of your own — daily prices from a free endpoint — and a two-tier fleet, with the comparison task routed to the stronger model. | ≤ $0.50 |

```bash
uv run python examples/01_offline_swarm.py
uv run python examples/02_live_graph.py

export ANTHROPIC_API_KEY=...
uv run python examples/03_ask.py
uv run python examples/04_report_and_resume.py            # `docs` extra for the PDF
uv sync --extra web
uv run python examples/05_news_digest.py "European AI regulation"
uv run python examples/06_market_brief.py NVDA MSFT ASML
```

Every one of them runs behind `myriapod.progress`, the live status line, so a
long planner turn looks like work rather than like a hang:

```
⠸ Swarming…   working      24 tasks  ▶   8  ✔    6  ·   10  ✖  0    41.2s  $0.0231
```

The costs above are the ceilings those files set, not estimates; a run that
reaches one stops and returns what it already has.
