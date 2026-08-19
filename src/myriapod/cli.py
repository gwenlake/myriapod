"""myriapod CLI.

``myriapod ask "question"``   — run a real swarm on a question.
``myriapod bench -n 1000``    — exercise the scheduler with simulated
                                agents, offline, no API keys.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from myriapod import __version__
from myriapod.core.scheduler import Swarm, SwarmResult
from myriapod.progress import arun_with_progress

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Planner/worker agent swarms: a frontier model plans, cheap fleets execute.",
)
console = Console(stderr=True)

DEFAULT_PLANNER = os.environ.get("MYRIAPOD_PLANNER", "anthropic:claude-opus-5")
DEFAULT_WORKER = os.environ.get("MYRIAPOD_WORKER", "anthropic:claude-haiku-4-5")

#: Hard $ budget applied unless ``--max-cost`` says otherwise. Cheap enough
#: that a mistyped question cannot burn a fortune, generous enough for a
#: real multi-section report; ``--max-cost 0`` removes the ceiling.
DEFAULT_MAX_COST = 1.5

#: Public list prices, $ per million tokens (input/output), from
#: https://platform.claude.com/docs/en/about-claude/pricing. Used to bill a
#: run automatically when the model is a known one, so ``--planner-price`` /
#: ``--worker-price`` are only needed for other providers or negotiated
#: rates. Keys are matched as substrings of the model string, so dated ids
#: (``claude-haiku-4-5-20251001``) and provider prefixes
#: (``anthropic:``, ``bedrock:anthropic.``) all resolve.
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-fable-5": {"input": 10.0, "output": 50.0},
    "claude-mythos-5": {"input": 10.0, "output": 50.0},
    "claude-opus-5": {"input": 5.0, "output": 25.0},
    "claude-opus-4-8": {"input": 5.0, "output": 25.0},
    "claude-opus-4-7": {"input": 5.0, "output": 25.0},
    "claude-opus-4-6": {"input": 5.0, "output": 25.0},
    "claude-opus-4-5": {"input": 5.0, "output": 25.0},
    # Sonnet 5 runs at introductory $2/$10 through 2026-08-31; the standard
    # rate below never under-reports a run.
    "claude-sonnet-5": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
}

#: Output ceiling per planner turn / worker answer, unless ``--max-tokens``.
DEFAULT_MAX_TOKENS = 8192

#: Raised default when ``--fanout`` asks for a plan the ceiling above cannot
#: hold: one brief is three to six sentences, so fifty of them in a single
#: ``plan_tasks`` call runs past 8192 output tokens and the plan comes back
#: truncated.
WIDE_PLAN_MAX_TOKENS = 16000

#: Beyond this many tasks, a plan needs more room than DEFAULT_MAX_TOKENS.
WIDE_PLAN_THRESHOLD = 20

#: Handed to the first planner turn as a note — never mixed into the goal:
#: the goal is what the root node shows and what titles the report, and a
#: planning instruction has no business in either. It states the run's actual
#: parameters rather than generic advice, because the two numbers that shape
#: a plan (how wide the fleet is, how wide the fan-out should be) are exactly
#: the two the planner cannot otherwise know.
def _planning_note(fanout: Optional[int], concurrency: int) -> Optional[str]:
    """The first-turn note built from the run's own flags."""
    bits = []
    if concurrency:
        bits.append(
            f"--concurrency {concurrency}: up to {concurrency} workers run at "
            f"the same time, so independent tasks cost almost nothing in wall "
            f"clock up to that width. Below it, breadth is free; above it, "
            f"tasks queue."
        )
    if fanout:
        bits.append(
            f"--fanout {fanout}: decompose into roughly {fanout} independent "
            f"tasks — wide and flat rather than deep — plus a final synthesis "
            f"that depends on them, batching across several plan_tasks calls. "
            f"Each brief must own a distinct, non-overlapping slice. If the "
            f"goal cannot honestly support that many, plan fewer: overlapping "
            f"briefs cost as much as real ones and produce duplicate "
            f"paragraphs."
        )
    if not bits:
        return None
    return "Run parameters:\n" + "\n".join("- " + b for b in bits)


async def _run_with_progress(
    swarm: Swarm,
    question: str,
    quiet: bool,
    planner_note: str | None = None,
    resume: dict | None = None,
) -> SwarmResult:
    """The shared status line (``myriapod.progress``), on this console."""
    return await arun_with_progress(
        swarm, question, planner_note=planner_note, resume=resume,
        quiet=quiet, console=console,
    )


#: Why a run can come back without any answer, and what to do about it.
_NO_ANSWER_HINTS = {
    "planner_stalled": "the planner answered in prose instead of calling a tool "
                       "twice in a row — check that the planner model supports "
                       "tool calling, or try another one with -p",
    "max_planner_turns": "the planner ran out of turns before calling finish — "
                         "raise --max-turns",
    "max_cost": "the run hit --max-cost before producing anything — raise the "
                "budget. A wide --fanout is planner-heavy: writing fifty briefs "
                "costs on the order of a dollar or two on a frontier planner, "
                "before a single worker runs",
    "max_tasks": "the planner blew past --max-tasks",
    "planner_error": "the planner turn kept failing (see the log entries) — "
                     "usually malformed tool arguments from the planner model",
}


def _explain_no_answer(result: SwarmResult) -> None:
    """A run with no content is a failure mode: never let it pass silently."""
    hint = _NO_ANSWER_HINTS.get(result.reason)
    console.print(
        f"[red]no answer produced[/red] (reason: {result.reason}, "
        f"{result.tasks_done} tasks done, {result.tasks_failed} failed)"
        + (f"\n[dim]{hint}[/dim]" if hint else "")
    )


def _cost_summary(result: SwarmResult) -> str:
    c = result.costs
    tokens = (
        c.planner_input_tokens + c.planner_output_tokens
        + c.workers_input_tokens + c.workers_output_tokens
    )
    if not tokens:
        return "cost unknown (the provider reported no token usage)"
    if not c.total_cost:
        return (f"{tokens:,} tokens (no list price known for these models: "
                f"pass --planner-price/--worker-price)")
    return f"${c.total_cost:.4f} (planner {c.planner_share:.0%})"


def _report_meta(result: SwarmResult, planner: str, worker: str) -> list[tuple[str, str]]:
    """Provenance recorded in the exported file."""
    return [
        ("Generated by", f"myriapod {__version__} — planner {planner}, workers {worker}"),
        ("Run", f"{result.reason} in {result.duration:.1f}s, "
                f"{result.tasks_done} tasks, {result.planner_turns} planner turns"),
        ("Cost", _cost_summary(result)),
    ]


def _save_outputs(
    result: SwarmResult,
    outputs: list[Path],
    question: str,
    planner: str,
    worker: str,
) -> None:
    """Write the answer to each path; the suffix picks the format."""
    from myriapod.report import ReportError, save_report

    if not result.content:
        console.print(
            "[yellow]nothing to write: the run produced no answer[/yellow]"
        )
        return
    meta = _report_meta(result, planner, worker)
    for path in outputs:
        try:
            save_report(path, result.content, title=question, meta=meta)
        except ReportError as e:
            console.print(f"[red]error:[/red] {e}")
            raise typer.Exit(2)
        console.print(f"[bold green]wrote[/] {path}")


def _print_result(
    result: SwarmResult,
    as_json: bool,
    save_tree: Optional[Path],
    quiet_content: bool = False,
) -> None:
    if save_tree:
        save_tree.parent.mkdir(parents=True, exist_ok=True)
        save_tree.write_text(
            json.dumps(result.tree, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        console.print(f"[dim]task tree saved to {save_tree}[/dim]")
    if as_json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        if not result.content:
            _explain_no_answer(result)
        return
    if not result.content:
        _explain_no_answer(result)
    elif not quiet_content:
        print(result.content)
    console.print(
        f"[bold {'green' if result.finished else 'yellow'}]"
        f"{result.reason}[/] in {result.duration:.1f}s — "
        f"{result.tasks_done} tasks done, {result.tasks_failed} failed, "
        f"{result.planner_turns} planner turns — {_cost_summary(result)}"
    )


def _model_pricing(model: str) -> Optional[dict[str, float]]:
    """List price for a known model string, or ``None`` if we don't know it.

    Longest key first, so ``claude-opus-4-5`` never matches on a shorter,
    ambiguous prefix.
    """
    name = model.lower()
    for key in sorted(MODEL_PRICING, key=len, reverse=True):
        if key in name:
            return dict(MODEL_PRICING[key])
    return None


def _parse_pricing(value: Optional[str], what: str) -> Optional[dict[str, float]]:
    """``"1.25/10"`` -> ``{"input": 1.25, "output": 10.0}`` ($ per M tokens)."""
    if not value:
        return None
    try:
        raw_in, raw_out = value.split("/")
        return {"input": float(raw_in), "output": float(raw_out)}
    except ValueError:
        console.print(
            f"[red]error:[/red] --{what}-price expects INPUT/OUTPUT in $ per "
            f"million tokens, e.g. '1.25/10'; got {value!r}"
        )
        raise typer.Exit(2)



def _start_viz(swarm: Swarm, port: int, open_browser: bool):
    from myriapod.viz import VizServer
    import webbrowser

    server = VizServer(lambda: swarm.tree, port=port)
    url = server.start()
    console.print(f"[bold magenta]viz[/] live graph at [link={url}]{url}[/link]")
    if open_browser:
        webbrowser.open(url)
    return server


def _viz_hold(server) -> None:
    server.mark_ended()
    console.print("[dim]viz still serving — Ctrl+C to quit[/dim]")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question or goal for the swarm."),
    planner: str = typer.Option(
        DEFAULT_PLANNER, "--planner", "-p",
        help="Planner model (Pydantic AI string, e.g. 'openai:gpt-5'). "
             "Env: MYRIAPOD_PLANNER.",
    ),
    worker: str = typer.Option(
        DEFAULT_WORKER, "--worker", "-w",
        help="Worker model (e.g. 'anthropic:claude-haiku-4-5'). "
             "Env: MYRIAPOD_WORKER.",
    ),
    strong_worker: Optional[str] = typer.Option(
        None, "--strong-worker",
        help="Model for the tasks the planner marks as high effort "
             "(synthesis, arbitration). Without it every task runs on "
             "--worker. Billed at its own list price.",
    ),
    reviewer: Optional[str] = typer.Option(
        None, "--reviewer",
        help="Turn on the quality gate: this model reads each finished "
             "answer against its brief and can send it back for one rewrite. "
             "Use a cheap model — it roughly doubles the calls per task.",
    ),
    concurrency: int = typer.Option(8, "--concurrency", "-c", min=1, max=2000,
                                    help="Max workers running at the same time. "
                                         "Size it to your provider rate limits."),
    fanout: Optional[int] = typer.Option(
        None, "--fanout", "-f", min=1, max=2000,
        help="Ask the planner for roughly this many parallel tasks (one "
             "worker each). Optional: without it the planner sizes the tree "
             "from the question itself. It stays a request — a goal that "
             "cannot support the number gets fewer tasks, not padded ones.",
    ),
    max_cost: Optional[float] = typer.Option(
        DEFAULT_MAX_COST, "--max-cost",
        help=f"Hard $ budget for the run (default ${DEFAULT_MAX_COST:g}); "
             "0 removes the ceiling. Enforced from the model list price, or "
             "from --planner-price/--worker-price for unknown models.",
    ),
    planner_price: Optional[str] = typer.Option(
        None, "--planner-price",
        help="Override the planner list price: INPUT/OUTPUT in $ per million "
             "tokens, e.g. '1.25/10'.",
    ),
    worker_price: Optional[str] = typer.Option(
        None, "--worker-price",
        help="Override the worker list price: INPUT/OUTPUT in $ per million "
             "tokens, e.g. '1/5'.",
    ),
    max_turns: int = typer.Option(12, "--max-turns", help="Max planner turns."),
    max_tasks: int = typer.Option(2000, "--max-tasks", help="Max tasks in the tree."),
    depth: int = typer.Option(3, "--depth", help="Max tree depth."),
    timeout: Optional[float] = typer.Option(300.0, "--timeout",
                                            help="Per-worker timeout (s)."),
    max_tokens: int = typer.Option(
        DEFAULT_MAX_TOKENS, "--max-tokens",
        help="Output-token ceiling per planner turn and per worker answer. "
             "Raise it for long deliverables; a truncated answer is reported "
             "as a missing <summary> block. Raised automatically for a wide "
             f"--fanout (> {WIDE_PLAN_THRESHOLD} tasks) if left at the default.",
    ),
    web: bool = typer.Option(False, "--web", help="Give workers web search "
                             "(DuckDuckGo; requires the [web] extra)."),
    output: Optional[list[Path]] = typer.Option(
        None, "--output", "-o",
        help="Write the answer to a file; the extension picks the format "
             "(.md, .txt, .pdf, .pptx — the last two need the [docs] extra). "
             "Repeatable: -o report.pdf -o deck.pptx.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the full result as JSON."),
    save_tree: Optional[Path] = typer.Option(None, "--save-tree",
                                             help="Write the task tree JSON here."),
    resume: Optional[Path] = typer.Option(
        None, "--resume",
        help="Continue a run from a tree written by --save-tree. Finished "
             "tasks keep their output and are not paid for twice; whatever "
             "was in flight when the run stopped goes back to pending. Use "
             "it when a run hit --max-cost or was interrupted — the tree is "
             "the only place the full worker outputs live.",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="No live progress."),
    viz: bool = typer.Option(False, "--viz", help="Live task-graph in the browser."),
    viz_port: int = typer.Option(8400, "--viz-port", help="Viz server port."),
    viz_open: bool = typer.Option(True, "--viz-open/--no-viz-open",
                                  help="Open the browser automatically."),
    viz_hold: bool = typer.Option(False, "--viz-hold/--no-viz-hold",
                                  help="Keep the viz server alive after the run "
                                       "(blocks until Ctrl+C). Off by default: a "
                                       "finished run should give you your shell "
                                       "back."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Scheduler logs."),
) -> None:
    """Run a swarm on a question: PLANNER plans, a fleet of WORKER executes."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    # Read the tree before anything is spent: a --resume pointing at a file
    # that does not parse must not be discovered after a fresh run has been
    # planned and paid for.
    previous = None
    if resume is not None:
        try:
            previous = json.loads(resume.read_text())
            done = sum(
                1 for n in previous["nodes"].values() if n["status"] == "done"
            )
        except (OSError, ValueError, KeyError, TypeError) as e:
            console.print(
                f"[red]error:[/red] cannot resume from {resume}: {e}. "
                "It must be a tree written by --save-tree."
            )
            raise typer.Exit(2)
        console.print(f"[dim]resuming from {resume}: {done} task(s) already done[/dim]")

    outputs = list(output or [])
    if outputs:
        # Fail before spending a whole run on an output we cannot write.
        from myriapod.report import FORMATS

        for path in outputs:
            if path.suffix.lower() not in FORMATS:
                what = path.suffix or "a file with no extension"
                console.print(
                    f"[red]error:[/red] cannot write {what}. "
                    f"Supported: {', '.join(FORMATS)}."
                )
                raise typer.Exit(2)

    from myriapod.adapters.pydantic_ai import build_swarm

    worker_tools = []
    if web:
        try:
            from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

            worker_tools.append(duckduckgo_search_tool())
        except ImportError:
            console.print(
                "[red]--web requires the web extra: uv add 'myriapod[web]'[/red]"
            )
            raise typer.Exit(2)

    # Explicit flags win; otherwise bill known models at their list price so
    # that --max-cost works out of the box.
    planner_pricing = _parse_pricing(planner_price, "planner") or _model_pricing(planner)
    worker_pricing = _parse_pricing(worker_price, "worker") or _model_pricing(worker)
    worker_models: str | dict[str, str] = worker
    if strong_worker:
        worker_models = {"low": worker, "standard": worker, "high": strong_worker}
        # One rate per tier, or the run bills its Opus tasks as Haiku and
        # --max-cost stops meaning anything.
        strong_pricing = _model_pricing(strong_worker)
        if worker_pricing and strong_pricing:
            worker_pricing = {
                "low": worker_pricing,
                "standard": worker_pricing,
                "high": strong_pricing,
            }
        elif worker_pricing and not strong_pricing:
            console.print(
                f"[yellow]warning:[/yellow] no list price known for "
                f"{strong_worker!r}; high-effort tasks are billed at the "
                f"{worker!r} rate, so the run is under-reported."
            )
    if max_cost is not None and max_cost <= 0:
        max_cost = None  # explicit opt-out: run without a budget ceiling

    if fanout and fanout > WIDE_PLAN_THRESHOLD and max_tokens == DEFAULT_MAX_TOKENS:
        # A wide plan is written in one planner turn; the default ceiling
        # cuts it off mid-brief, which looks like a small plan rather than
        # a failure. Only override a ceiling the user did not choose.
        max_tokens = WIDE_PLAN_MAX_TOKENS
        console.print(
            f"[dim]--fanout {fanout}: raising --max-tokens to {max_tokens} "
            f"so the plan is not truncated[/dim]"
        )
    # Planning is not free: each brief is a few hundred output tokens on the
    # frontier model, and a wide plan can eat a small budget before a worker
    # runs at all. Measured around $0.03 per brief on Opus 5.
    if fanout and fanout >= 25 and max_cost is not None:
        est = 0.03 * fanout
        if est > max_cost / 2:
            console.print(
                f"[yellow]warning:[/yellow] planning {fanout} tasks costs roughly "
                f"${est:.2f} on the planner alone, against a ${max_cost:.2f} budget. "
                f"Raise --max-cost (or lower --fanout) or the run may stop before "
                f"the workers deliver."
            )
    if fanout and fanout > max_tasks:
        console.print(
            f"[yellow]warning:[/yellow] --fanout {fanout} exceeds --max-tasks "
            f"{max_tasks}; the tree will be capped there."
        )
    if max_cost is not None and not (planner_pricing or worker_pricing):
        console.print(
            "[yellow]warning:[/yellow] --max-cost cannot be enforced without "
            f"pricing, and no list price is known for {planner!r}/{worker!r}; "
            "add --planner-price and --worker-price (e.g. '1.25/10')."
        )

    swarm = build_swarm(
        planner_model=planner,
        worker_model=worker_models,
        reviewer_model=reviewer,
        reviewer_pricing=_model_pricing(reviewer) if reviewer else None,
        worker_tools=worker_tools,
        max_concurrency=concurrency,
        max_planner_turns=max_turns,
        max_tasks=max_tasks,
        max_depth=depth,
        max_cost=max_cost,
        worker_timeout=timeout,
        planner_max_tokens=max_tokens,
        worker_max_tokens=max_tokens,
        planner_pricing=planner_pricing,
        worker_pricing=worker_pricing,
    )
    server = _start_viz(swarm, viz_port, viz_open) if viz else None
    try:
        # The goal stays the user's own words — it is what the root node
        # shows and what titles the report. --fanout rides along as a note to
        # the first planner turn instead.
        result = asyncio.run(
            _run_with_progress(swarm, question, quiet,
                               _planning_note(fanout, concurrency), previous)
        )
    except Exception as e:  # missing API key, bad model string...
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(1)
    _print_result(result, as_json, save_tree, quiet_content=bool(outputs))
    if outputs:
        _save_outputs(result, outputs, question, planner, worker)
    if server is not None and viz_hold:
        _viz_hold(server)
    raise typer.Exit(0 if result.finished else 3)


@app.command()
def bench(
    n: int = typer.Option(100, "--tasks", "-n", min=1, max=100_000,
                          help="Number of simulated tasks."),
    concurrency: int = typer.Option(200, "--concurrency", "-c", min=1, max=10_000),
    delay: float = typer.Option(0.05, "--delay", help="Simulated work (s) per task."),
    jitter: float = typer.Option(0.05, "--jitter", help="Extra random delay (s)."),
    batch: int = typer.Option(50, "--batch", help="Subtasks per plan_tasks call."),
    waves: int = typer.Option(
        1, "--waves", min=1,
        help="Spread the tasks over this many planner turns. Each wave runs "
             "only once the previous one is done, which is what the sequence "
             "of planner nodes on the right of the graph shows.",
    ),
    planner_delay: float = typer.Option(
        0.0, "--planner-delay",
        help="Seconds each simulated planner turn spends 'thinking'. Give it "
             "a second or two to watch the root go amber and the workers "
             "carry on regardless.",
    ),
    groups: int = typer.Option(
        0, "--groups", min=0,
        help="Add this many synthesis tasks, each depending on a slice of the "
             "leaves. Dependencies are what the graph colours by group, so "
             "raise it when looking at --viz; leave it at 0 when timing.",
    ),
    viz: bool = typer.Option(False, "--viz", help="Live task-graph in the browser."),
    viz_port: int = typer.Option(8400, "--viz-port", help="Viz server port."),
    viz_open: bool = typer.Option(True, "--viz-open/--no-viz-open",
                                  help="Open the browser automatically."),
    viz_hold: bool = typer.Option(False, "--viz-hold/--no-viz-hold",
                                  help="Keep the viz server alive after the run "
                                       "(blocks until Ctrl+C)."),
) -> None:
    """Scheduler benchmark with simulated agents — offline, free, no keys.

    Proves the swarm can drive hundreds or thousands of concurrent workers:
    prints wall time, throughput, and the max parallelism actually reached.
    """
    from myriapod.testing import Recorder, SimWorker, fanout_turns, scripted_planner_factory

    recorder = Recorder()
    swarm = Swarm(
        planner_factory=scripted_planner_factory(
            fanout_turns(n, batch=batch, groups=groups, waves=waves),
            delay=planner_delay,
        ),
        worker_factory=lambda node: SimWorker(node, recorder, delay=delay, jitter=jitter),
        max_concurrency=concurrency,
        max_tasks=max(2000, n + groups + 10),
        max_planner_turns=waves + 3,
        worker_pricing={"input": 1.0, "output": 5.0},
    )
    server = _start_viz(swarm, viz_port, viz_open) if viz else None
    started = time.time()
    result = asyncio.run(_run_with_progress(swarm, f"bench {n} tasks", quiet=False))
    wall = time.time() - started
    ideal = (delay + jitter / 2) * (n / concurrency)
    console.print(
        f"[bold green]bench[/]: {result.tasks_done}/{n} tasks in {wall:.2f}s "
        f"→ [bold]{result.tasks_done / wall:,.0f} tasks/s[/] — "
        f"max parallel {recorder.max_parallel} (asked {concurrency}) — "
        f"ideal compute time ≈ {ideal:.2f}s"
    )
    expected = n + groups
    if result.tasks_done != expected:
        console.print(f"[red]expected {n} done, got {result.tasks_done}[/red]")
        raise typer.Exit(1)
    if server is not None and viz_hold:
        _viz_hold(server)


@app.command()
def version() -> None:
    """Print the myriapod version."""
    print(__version__)


if __name__ == "__main__":
    app()
