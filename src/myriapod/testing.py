"""Test and benchmark doubles: run a full swarm without any LLM.

Used by the test suite and by ``myriapod bench``, which exercises the
scheduler with hundreds or thousands of simulated workers to validate
throughput and correctness offline — no API keys, no cost.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field

from myriapod.core.protocol import RunOutcome
from myriapod.core.task_tree import TaskNode


class ScriptedPlanner:
    """Planner double: each turn executes a list of (tool_name, kwargs).

    ``delay`` keeps the turn "thinking" after its tool calls have landed —
    the shape of a real planner writing a long plan. The scheduler is
    supposed to dispatch the tasks it created without waiting for the turn
    to return, so tests can assert on ``returned_at`` versus worker starts.
    """

    def __init__(self, turns: list[list[tuple[str, dict]]], delay: float = 0.0):
        self.turns = list(turns)
        self.delay = delay
        self.tools: list = []
        self.name = "scripted-planner"
        self.messages: list[str] = []
        self.returned_at: list[float] = []

    def _call(self, name: str, kwargs: dict) -> str:
        fn = next(t for t in self.tools if getattr(t, "__name__", None) == name)
        return fn(**kwargs)

    async def run(self, message: str, context: str | None = None) -> RunOutcome:
        self.messages.append(message)
        calls = self.turns.pop(0) if self.turns else []
        for tool_name, kwargs in calls:
            self._call(tool_name, kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        self.returned_at.append(time.monotonic())
        return RunOutcome(content="ok", input_tokens=1000, output_tokens=200)


def scripted_planner_factory(turns: list[list[tuple[str, dict]]], delay: float = 0.0):
    """PlannerFactory for a :class:`ScriptedPlanner` (captures the tools)."""

    planners: list[ScriptedPlanner] = []

    def factory(tools: list) -> ScriptedPlanner:
        planner = ScriptedPlanner(turns, delay=delay)
        planner.tools = tools
        planners.append(planner)
        return planner

    factory.planners = planners  # type: ignore[attr-defined]
    return factory


@dataclass
class Recorder:
    """Shared instrumentation across simulated workers."""

    current: int = 0
    max_parallel: int = 0
    started: dict[str, float] = field(default_factory=dict)
    ended: dict[str, float] = field(default_factory=dict)


class SimWorker:
    """Simulated worker: sleeps, then answers with a summary block."""

    def __init__(
        self,
        node: TaskNode,
        recorder: Recorder,
        delay: float = 0.05,
        jitter: float = 0.0,
        fail_first: set[str] | frozenset[str] = frozenset(),
        sleep_forever: set[str] | frozenset[str] = frozenset(),
    ):
        self.node = node
        self.recorder = recorder
        self.delay = delay
        self.jitter = jitter
        self.fail_first = fail_first
        self.sleep_forever = sleep_forever
        self.name = f"sim-worker-{node.id}"

    async def run(self, message: str, context: str | None = None) -> RunOutcome:
        r = self.recorder
        r.current += 1
        r.max_parallel = max(r.max_parallel, r.current)
        r.started.setdefault(self.node.id, time.monotonic())
        try:
            if self.node.id in self.sleep_forever:
                await asyncio.sleep(3600)
            delay = self.delay + (random.random() * self.jitter if self.jitter else 0)
            await asyncio.sleep(delay)
            if self.node.id in self.fail_first and self.node.attempts <= 1:
                raise RuntimeError("transient failure")
            content = (
                f"FULL RESULT of {self.node.id} (deps: {bool(context)})\n"
                f"<summary>Summary of {self.node.id}.</summary>"
            )
            return RunOutcome(content=content, input_tokens=200, output_tokens=100)
        finally:
            r.ended[self.node.id] = time.monotonic()
            r.current -= 1


class SimReviewer:
    """Reviewer double: rejects the ids in ``reject`` once, then passes.

    Mirrors the real thing closely enough to test the loop — a first answer
    turned back with reasons, a rewrite accepted — without a model in sight.
    """

    def __init__(
        self,
        node: TaskNode,
        reject: set[str] | frozenset[str] = frozenset(),
        rounds: int = 1,
        verdict: str | None = None,
    ):
        self.node = node
        self.reject = reject
        self.rounds = rounds
        self.verdict = verdict
        self.name = f"sim-reviewer-{node.id}"
        self.seen: list[str] = []

    async def run(self, message: str, context: str | None = None) -> RunOutcome:
        self.seen.append(context or "")
        if self.verdict is not None:
            content = self.verdict
        elif self.node.id in self.reject and _reviews_of(self.node.id) < self.rounds:
            content = (
                "VERDICT: REVISE\n1. Too shallow: name the actors and give figures."
            )
        else:
            content = "VERDICT: PASS"
        _count_review(self.node.id)
        return RunOutcome(content=content, input_tokens=50, output_tokens=20)


#: Reviews already delivered per task id, so a double can reject "the first
#: answer only" across the separate reviewer instances a factory hands out.
_REVIEW_COUNTS: dict[str, int] = {}


def _reviews_of(task_id: str) -> int:
    return _REVIEW_COUNTS.get(task_id, 0)


def _count_review(task_id: str) -> None:
    _REVIEW_COUNTS[task_id] = _REVIEW_COUNTS.get(task_id, 0) + 1


def reset_review_counts() -> None:
    """Call between tests: the counters are module-level by design."""
    _REVIEW_COUNTS.clear()


def fanout_turns(
    n_tasks: int, batch: int = 50, groups: int = 0, waves: int = 1
) -> list[list[tuple[str, dict]]]:
    """Planner script: fan out ``n_tasks`` leaves, then finish.

    Mirrors what a real planner does on a large map-style goal: several
    plan_tasks calls per turn, one finish turn at the end.

    ``groups`` adds that many synthesis tasks, each depending on a slice of
    the leaves — the shape of a real report (sections plus the task that
    reconciles them). They are the only thing that exercises dependency
    edges, so keep them at zero when timing the scheduler and raise them
    when looking at the graph.

    ``waves`` spreads the leaves over that many planner turns instead of
    one. The scheduler only consults the planner when the frontier is
    exhausted, so the waves run strictly one after another — which is how a
    real re-plan behaves, and the only way to exercise the sequence of
    planner deliberations offline.
    """
    waves = max(1, waves)
    per_wave = [n_tasks // waves] * waves
    for i in range(n_tasks - sum(per_wave)):
        per_wave[i] += 1

    turns: list[list[tuple[str, dict]]] = []
    made = 0
    for w, count in enumerate(per_wave):
        turn: list[tuple[str, dict]] = []
        remaining = count
        while remaining > 0:
            size = min(batch, remaining)
            subtasks = [
                {"description": f"simulated task #{made + j + 1}"}
                for j in range(size)
            ]
            turn.append(
                ("plan_tasks", {"parent_id": "root", "subtasks": json.dumps(subtasks)})
            )
            made += size
            remaining -= size
        # Synthesis tasks ride on the last wave: they depend on the leaves,
        # so they can only be planned once those exist.
        if groups > 0 and w == waves - 1:
            span = max(1, n_tasks // groups)
            synth = []
            for g in range(groups):
                start = g * span + 1
                end = min(n_tasks, start + span - 1)
                if start > n_tasks:
                    break
                synth.append({
                    "description": f"synthesis #{g + 1} of tasks {start}-{end}",
                    "depends_on": [str(k) for k in range(start, end + 1)],
                })
            if synth:
                turn.append(
                    ("plan_tasks", {"parent_id": "root", "subtasks": json.dumps(synth)})
                )
        turns.append(turn)

    turns.append([("finish", {"final_answer": f"{n_tasks} tasks completed."})])
    return turns
