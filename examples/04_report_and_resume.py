"""A run that produces a document — and that survives being interrupted.

Two things worth copying from this file:

1. **Save the tree.** The task tree is the only place full worker outputs
   live; a process that exits without writing it has destroyed them. Hence
   the `finally`. `resume=` then rebuilds the run from that file: finished
   tasks keep their answers and are never paid for twice, whatever was in
   flight goes back to pending. A run stopped by `max_cost` is finished for
   a few cents instead of being redone — so kill this one halfway and start
   it again to watch it pick up where it stopped.

2. **Render the answer.** Workers reply in Markdown, so `save_report` parses
   it once and renders per format. `.md`/`.txt` need nothing; `.pdf`/`.pptx`
   need the `docs` extra (`uv sync --extra docs`).

    export ANTHROPIC_API_KEY=...
    uv run python examples/04_report_and_resume.py
"""

import json
from pathlib import Path

from myriapod.adapters.pydantic_ai import build_swarm
from myriapod.progress import run_with_progress
from myriapod.report import save_report

GOAL = (
    "Write a 1200-word briefing on retrieval-augmented generation in 2026: "
    "what changed since 2024, where it still fails, and when a fine-tune or a "
    "long context window is the better answer. Include a comparison table."
)
STATE = Path("examples/out/report.tree.json")
PLANNER, WORKER = "anthropic:claude-sonnet-5", "anthropic:claude-haiku-4-5"


def main() -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    swarm = build_swarm(
        planner_model=PLANNER,
        worker_model=WORKER,
        max_concurrency=8,
        max_cost=0.60,
        worker_max_tokens=16000,     # sections are long; 8192 would cut them
        planner_pricing={"input": 3.0, "output": 15.0},
        worker_pricing={"input": 1.0, "output": 5.0},
    )

    # Anything already on disk is work we are not paying for again.
    resume = json.loads(STATE.read_text()) if STATE.exists() else None
    if resume:
        print(f"resuming from {STATE}")

    try:
        result = run_with_progress(swarm, GOAL, resume=resume)
    finally:
        # Even on a crash or a Ctrl-C: the tree is the run's only memory.
        STATE.write_text(json.dumps(swarm.tree.to_dict(), ensure_ascii=False))

    if not result.content:
        print(f"no answer: {result.reason} — run again to resume")
        return

    meta = [
        ("Planner", PLANNER),
        ("Workers", WORKER),
        ("Outcome", f"{result.reason} — {result.tasks_done} tasks"),
        ("Cost", f"${result.costs.total_cost:.4f}"),
    ]
    for out in ("examples/out/rag-2026.md", "examples/out/rag-2026.pdf"):
        print("written:", save_report(out, result.content, title=GOAL, meta=meta))

    print(f"\n{result.reason} — ${result.costs.total_cost:.4f}")


if __name__ == "__main__":
    main()
