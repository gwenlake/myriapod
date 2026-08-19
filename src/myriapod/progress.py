"""A live status line for a run without the graph.

Without `--viz` a swarm is silent: the planner can spend minutes writing a
fifty-task plan before the first worker starts, and a terminal that prints
nothing in that time is indistinguishable from one that has hung. So the
line always *moves* — a spinner and a rotating verb — and always says which
phase the run is in, even when no counter has changed yet:

    ⠹ Marching…   planning      0 tasks  ▶   0  ✔    0  ·    0  ✖  0    12.4s  $0.0000
    ⠸ Swarming…   working      24 tasks  ▶   8  ✔    6  ·   10  ✖  0    41.2s  $0.0231

The phase comes from the tree itself: the planner flags the root as running
for the duration of its turn (``mark_planner_thinking``), which is the only
signal that exists while a turn is in flight — token counts arrive when the
call returns, minutes later.

Non-interactive output (a pipe, a log file, `-q`) gets no line at all: a
spinner repainted into a file is noise, and the caller is not watching.
"""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.text import Text

#: Braille spinner: one glyph per frame, and it never stops turning.
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

#: Rotating verbs, purely so the eye can tell a live run from a frozen one
#: at a glance. They carry no information — the numbers beside them do.
VERBS = (
    "Scuttling", "Swarming", "Marching", "Skittering", "Foraging",
    "Burrowing", "Wriggling", "Bustling", "Tunnelling", "Rummaging",
)

#: Seconds each verb stays up.
VERB_SECONDS = 4.0

#: Repaints per second. Fast enough to look alive, slow enough to be free.
REFRESH = 8


def _phase(tree: Any) -> str:
    """What the run is doing right now, in one word."""
    turns = getattr(tree, "planner_turns", None)
    if turns and turns[-1].get("running"):
        return "planning"
    stats = tree.stats()
    if stats["by_status"].get("in_progress"):
        return "working"
    return "scheduling"


def status_text(swarm: Any, started: float, now: float | None = None) -> Text:
    """One line: spinner, verb, phase, counters, elapsed, cost."""
    now = time.time() if now is None else now
    elapsed = now - started
    frame = SPINNER[int(elapsed * REFRESH) % len(SPINNER)]
    verb = VERBS[int(elapsed / VERB_SECONDS) % len(VERBS)]

    # The verb is padded so the counters beside it never jump columns as it
    # rotates — a line that shifts is harder to read than one that does not.
    label = f"{frame} {verb + '…':<12}"

    tree = getattr(swarm, "tree", None)
    if tree is None:                      # before arun() builds it
        return Text(f"{label} starting up   {elapsed:5.1f}s", style="cyan")

    s = tree.stats()
    by = s["by_status"]
    line = (
        f"{label} {_phase(tree):<10} "
        f"{s['tasks']:>4} tasks  "
        f"▶ {by.get('in_progress', 0):>3}  "
        f"✔ {by.get('done', 0):>4}  "
        f"· {by.get('pending', 0):>4}  "
        f"✖ {by.get('failed', 0):>2}  "
        f"{elapsed:6.1f}s  ${s['cost']:.4f}"
    )
    return Text(line, style="cyan")


def _interactive(console: Console | None) -> bool:
    return sys.stderr.isatty() if console is None else console.is_terminal


async def arun_with_progress(
    swarm: Any,
    goal: str,
    *,
    planner_note: str | None = None,
    resume: Any = None,
    quiet: bool = False,
    console: Console | None = None,
) -> Any:
    """``swarm.arun(...)`` with the status line above painted while it runs.

    The run is a task and the line is a poll of the tree, deliberately: the
    scheduler has no reporting hook and should not grow one for this — the
    tree already knows everything worth showing.
    """
    task = asyncio.create_task(swarm.arun(goal, planner_note, resume))
    if quiet or not _interactive(console):
        return await task

    console = console or Console(stderr=True)
    started = time.time()
    with Live(status_text(swarm, started), console=console, refresh_per_second=REFRESH) as live:
        while not task.done():
            live.update(status_text(swarm, started))
            await asyncio.sleep(1 / REFRESH)
        live.update(status_text(swarm, started))
    return task.result()


def run_with_progress(swarm: Any, goal: str, **kwargs: Any) -> Any:
    """Blocking :func:`arun_with_progress`, for scripts and notebooks."""
    return asyncio.run(arun_with_progress(swarm, goal, **kwargs))
