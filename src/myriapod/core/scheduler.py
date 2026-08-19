"""myriapod scheduler: one frontier planner driving a fleet of cheap workers.

Framework-free: the scheduler only knows :class:`AgentLike` (see
``myriapod.core.protocol``). Adapters supply the factories.

Scheduling model:

- **Continuous frontier dispatch.** Ready leaves are dispatched up to
  ``max_concurrency``; the loop wakes on the *first* completion
  (``asyncio.wait(FIRST_COMPLETED)``) and immediately dispatches whatever
  that completion unblocked. Workers never wait for a batch. asyncio
  handles thousands of concurrent workers on one loop.
- **The planner is only consulted when the frontier is exhausted** —
  start, blocked, or done. Planner turns are the expensive resource.
- **Failures are absorbed before they reach the planner.** A failed leaf
  is auto-retried (``auto_retry``) with exponential backoff inside the
  worker coroutine. Only persistent failures surface at a planner turn.
- **Context isolation.** The planner sees ``tree.render()`` (collapsed and
  capped on big runs); each worker sees its task plus the full outputs of
  its direct dependencies.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from myriapod.core.planner import (
    SwarmState,
    make_planner_tools,
    planner_turn_message,
)
from myriapod.core.protocol import PlannerFactory, WorkerFactory
from myriapod.core.task_tree import TaskTree
from myriapod.core.worker import (
    DEFAULT_CONTEXT_CHARS,
    ReviewPolicy,
    compute_cost,
    resolve_pricing,
    run_worker,
)

logger = logging.getLogger("myriapod")

#: While a planner turn is in flight, the tasks it creates appear in the tree
#: one plan_tasks call at a time, with nothing to wake the loop — so it polls
#: the frontier at this interval instead of blocking until the turn returns.
#: Only paid during a planner turn, and cheap next to a model round-trip.
PLANNER_POLL_SECONDS = 0.1


@dataclass
class SwarmCosts:
    """Token and dollar breakdown by role."""

    planner_input_tokens: int = 0
    planner_output_tokens: int = 0
    planner_cost: float = 0.0
    workers_input_tokens: int = 0
    workers_output_tokens: int = 0
    workers_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        return self.planner_cost + self.workers_cost

    @property
    def planner_share(self) -> float:
        total = self.total_cost
        return self.planner_cost / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "planner": {
                "input_tokens": self.planner_input_tokens,
                "output_tokens": self.planner_output_tokens,
                "cost": round(self.planner_cost, 6),
            },
            "workers": {
                "input_tokens": self.workers_input_tokens,
                "output_tokens": self.workers_output_tokens,
                "cost": round(self.workers_cost, 6),
            },
            "total_cost": round(self.total_cost, 6),
            "planner_share": round(self.planner_share, 4),
        }


@dataclass
class SwarmResult:
    """Outcome of one swarm run."""

    content: str | None
    finished: bool
    reason: str
    tree: dict[str, Any]
    costs: SwarmCosts
    planner_turns: int
    tasks_done: int
    tasks_failed: int
    duration: float
    log: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "finished": self.finished,
            "reason": self.reason,
            "planner_turns": self.planner_turns,
            "tasks_done": self.tasks_done,
            "tasks_failed": self.tasks_failed,
            "duration": round(self.duration, 3),
            "costs": self.costs.to_dict(),
            "log": self.log,
            "tree": self.tree,
        }


@dataclass
class Swarm:
    """Planner/worker swarm over a shared :class:`TaskTree`.

    ``planner_factory`` receives the tree-piloting tool functions for one
    run and returns the planner agent. ``worker_factory`` returns the agent
    for each leaf task (may be a shared instance). Both come from an
    adapter (``myriapod.adapters.*``) or from your own code.
    """

    planner_factory: PlannerFactory
    worker_factory: WorkerFactory
    planner_pricing: Optional[dict[str, float]] = None
    #: ``{"input": .., "output": ..}`` in $ per million tokens — or one such
    #: table per effort tier (``{"high": {...}, "standard": {...}}``) when
    #: the fleet routes tiers to different models.
    worker_pricing: Optional[dict[str, Any]] = None

    #: Optional quality gate: builds the judge that reads a finished answer
    #: against its brief before it lands in the tree (see ``core.reviewer``).
    #: ``None`` — the default — keeps the base loop, one call per task.
    reviewer_factory: Optional[WorkerFactory] = None
    reviewer_pricing: Optional[dict[str, float]] = None
    #: Rewrites allowed per task when the reviewer rejects the answer.
    max_revisions: int = 1

    max_concurrency: int = 8
    max_planner_turns: int = 12
    #: Consecutive failed planner turns tolerated before giving up on the run.
    max_planner_errors: int = 3
    max_depth: int = 3
    max_tasks: int = 2000
    max_cost: float | None = None
    auto_retry: int = 1
    worker_timeout: float | None = None
    #: Characters of dependency output a single worker may be handed. Shared
    #: fairly between its dependencies rather than concatenated whole.
    worker_context_chars: int = DEFAULT_CONTEXT_CHARS
    #: Seconds a finishing planner turn is given to return before it is
    #: cancelled — long enough for the model to close its answer, short
    #: enough that a wedged turn cannot hold the run open.
    finish_grace: float = 30.0

    #: Introspection: the live tree of the current run (set by arun), so a
    #: UI or CLI can poll progress while the run is in flight.
    tree: TaskTree | None = None

    # ------------------------------------------------------------------ #

    def run(
        self,
        goal: str,
        planner_note: str | None = None,
        resume: TaskTree | dict[str, Any] | None = None,
    ) -> SwarmResult:
        """Synchronous wrapper around :meth:`arun`."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(goal, planner_note, resume))
        raise RuntimeError(
            "Swarm.run() cannot be called from a running event loop; "
            "use `await swarm.arun(goal)` instead."
        )

    async def arun(
        self,
        goal: str,
        planner_note: str | None = None,
        resume: TaskTree | dict[str, Any] | None = None,
    ) -> SwarmResult:
        """Run the swarm on ``goal``.

        ``planner_note`` is handed to the *first* planner turn only, for
        instructions about how to plan rather than what to achieve (the CLI's
        ``--fanout`` uses it). It deliberately does not touch the goal: the
        goal is what the root node shows, what titles the report, and what
        every later turn is judged against.

        ``resume`` restarts from a tree written by an earlier run (its
        ``SwarmResult.tree``, or a live :class:`TaskTree`). Finished tasks
        keep their output and are never re-run; tasks that were in flight
        when the run died go back to pending. The planner's first turn then
        sees a half-finished tree and continues from it — which is what makes
        a five-dollar run survivable when the network drops at task 40 of 50.
        """
        started = time.time()
        tree = self._resume_tree(resume, goal) if resume is not None else None
        if tree is None:
            tree = TaskTree(goal, max_depth=self.max_depth)
        self.tree = tree
        state = SwarmState()
        costs = SwarmCosts()
        planner = self.planner_factory(make_planner_tools(tree, state))
        review_policy = (
            ReviewPolicy(
                reviewer_factory=self.reviewer_factory,
                worker_factory=self.worker_factory,
                pricing=self.reviewer_pricing,
                max_revisions=self.max_revisions,
            )
            if self.reviewer_factory is not None and self.max_revisions > 0
            else None
        )

        running: dict[str, asyncio.Task[None]] = {}
        planner_task: asyncio.Task | None = None
        planner_turns = 0
        planner_errors = 0
        stalls = 0
        note: str | None = planner_note
        reason = "finished"

        turn_started = 0.0

        def book_turn(outcome: Any) -> None:
            """Account for one planner turn, wherever it was awaited."""
            costs.planner_input_tokens += getattr(outcome, "input_tokens", 0)
            costs.planner_output_tokens += getattr(outcome, "output_tokens", 0)
            costs.planner_cost = compute_cost(
                costs.planner_input_tokens,
                costs.planner_output_tokens,
                self.planner_pricing,
            )
            # The root node is where the planner's own spend and identity
            # live, so every surface reading the tree (live cost line, graph,
            # node panel) sees them.
            tree.record_planner_turn(
                getattr(planner, "name", "planner"),
                costs.planner_input_tokens,
                costs.planner_output_tokens,
                costs.planner_cost,
                time.time() - turn_started,
            )

        logger.info("swarm run started: %r", goal[:120])

        try:
            while True:
                if state.finished:
                    # finish() lands as a tool call *during* a turn, so the
                    # turn is usually still in flight here. Give it a moment
                    # to return: cancelling it would throw away the tokens it
                    # just spent, and the run would under-report its own cost.
                    if planner_task is not None:
                        try:
                            outcome = await asyncio.wait_for(
                                planner_task, timeout=self.finish_grace
                            )
                            book_turn(outcome)
                        except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                            logger.warning("planner turn did not return after finish")
                        planner_task = None
                    break

                ready = [n for n in tree.ready_tasks() if n.id not in running]

                # -- Planner turn: only when the frontier is exhausted. ---- #
                if planner_task is None and not running and not ready:
                    if tree.is_complete() and not note:
                        note = (
                            "All tasks are settled. Check the summaries against "
                            "the goal: if a part is missing, thin, or contradicted "
                            "by another, plan follow-up tasks. Otherwise call "
                            "finish (prefer from_task_id for a long deliverable)."
                        )
                    if planner_turns >= self.max_planner_turns:
                        reason = "max_planner_turns"
                        break
                    planner_turns += 1
                    state.planned_something = False
                    logger.info("planner turn %d", planner_turns)
                    # The turn runs *alongside* dispatch. Its plan_tasks calls
                    # mutate the tree as it makes them, so a worker starts on
                    # the first brief while the planner is still writing the
                    # fiftieth. Awaiting the whole turn instead left the fleet
                    # idle for as long as the planner took to write the plan —
                    # minutes, on a fifty-task fan-out.
                    turn_started = time.time()
                    tree.mark_planner_thinking(getattr(planner, "name", "planner"))
                    planner_task = asyncio.create_task(
                        planner.run(planner_turn_message(tree, note))
                    )
                    note = None
                    continue

                # -- Dispatch: fill the worker pool. ----------------------- #
                for node in ready:
                    if len(running) >= self.max_concurrency:
                        break
                    agent = self.worker_factory(node)
                    running[node.id] = asyncio.create_task(
                        run_worker(
                            agent,
                            tree,
                            node,
                            timeout=self.worker_timeout,
                            pricing=resolve_pricing(self.worker_pricing, node.effort),
                            context_chars=self.worker_context_chars,
                            review=review_policy,
                        )
                    )

                waiters: set[asyncio.Task] = set(running.values())
                if planner_task is not None:
                    waiters.add(planner_task)
                if not waiters:
                    continue  # frontier empty again -> planner turn

                # -- Wait for the first completion, then refill. ----------- #
                done, _ = await asyncio.wait(
                    waiters,
                    timeout=PLANNER_POLL_SECONDS if planner_task is not None else None,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    continue  # poll tick: pick up whatever the turn just planned

                # -- A finished planner turn: account for it, then re-plan. - #
                if planner_task is not None and planner_task in done:
                    turn, planner_task = planner_task, None
                    try:
                        outcome = turn.result()
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:  # noqa: BLE001
                        # A planner turn that blows up (bad tool arguments, a
                        # provider hiccup) must not take the run down with it:
                        # work already done still has to reach the caller.
                        planner_errors += 1
                        logger.warning("planner turn failed: %s: %s", type(e).__name__, e)
                        state.log.append(
                            {"action": "planner_error", "error": f"{type(e).__name__}: {e}"}
                        )
                        if planner_errors >= self.max_planner_errors:
                            reason = "planner_error"
                            break
                        note = (
                            f"Your previous turn failed with {type(e).__name__}: {e}. "
                            "Retry with a smaller, simpler tool call."
                        )
                    else:
                        planner_errors = 0
                        book_turn(outcome)
                        if len(tree) > self.max_tasks:
                            reason = "max_tasks"
                            logger.warning(
                                "task limit exceeded: %d > %d", len(tree), self.max_tasks
                            )
                            break
                        if not state.planned_something:
                            stalls += 1
                            if stalls >= 2:
                                reason = "planner_stalled"
                                break
                            note = (
                                "You made no valid tool call. Use plan_tasks, "
                                "retry_task, skip_task, or finish."
                            )
                        else:
                            stalls = 0
                    if self._over_budget(tree, costs):
                        reason = "max_cost"
                        break

                for task_id in [tid for tid, t in running.items() if t in done]:
                    running.pop(task_id)

                # -- Auto-retry fresh failures (backoff happens in-worker). - #
                for node in tree.failed_tasks():
                    if node.id not in running and node.attempts <= self.auto_retry:
                        logger.warning(
                            "auto-retry task %s (attempt %d): %s",
                            node.id,
                            node.attempts + 1,
                            node.error,
                        )
                        tree.reset_task(node.id)

                self._collect_worker_costs(tree, costs)
                if self._over_budget(tree, costs):
                    reason = "max_cost"
                    break
        finally:
            if planner_task is not None:
                planner_task.cancel()
                await asyncio.gather(planner_task, return_exceptions=True)
            if running:
                for task in running.values():
                    task.cancel()
                await asyncio.gather(*running.values(), return_exceptions=True)

        self._collect_worker_costs(tree, costs)
        stats = tree.stats()
        content = state.final_answer
        if content is None:
            content = self._fallback_answer(tree)
        result = SwarmResult(
            content=content,
            finished=state.finished,
            reason=reason,
            tree=tree.to_dict(),
            costs=costs,
            planner_turns=planner_turns,
            tasks_done=stats["by_status"].get("done", 0),
            tasks_failed=stats["by_status"].get("failed", 0),
            duration=time.time() - started,
            log=state.log,
        )
        logger.info(
            "swarm run %s in %.1fs — %d task(s) done, %d planner turn(s), "
            "cost $%.4f (planner share %.0f%%)",
            reason,
            result.duration,
            result.tasks_done,
            planner_turns,
            costs.total_cost,
            costs.planner_share * 100,
        )
        return result

    # ------------------------------------------------------------------ #

    def _resume_tree(
        self, resume: TaskTree | dict[str, Any], goal: str
    ) -> TaskTree:
        """Rebuild the tree of an interrupted run and make it runnable again."""
        tree = resume if isinstance(resume, TaskTree) else TaskTree.from_dict(resume)
        reopened = tree.reopen_running()
        if tree.goal != goal:
            # The tree's goal wins: it is what every completed task was
            # written against, and what the planner will be judged on.
            logger.warning(
                "resuming a tree whose goal differs from the one passed; "
                "keeping the tree's goal"
            )
        stats = tree.stats()
        logger.info(
            "resuming run %s: %d task(s), %s — %d reopened",
            tree.run_id,
            stats["tasks"],
            ", ".join(f"{v} {k}" for k, v in sorted(stats["by_status"].items())) or "none",
            len(reopened),
        )
        return tree

    def _collect_worker_costs(self, tree: TaskTree, costs: SwarmCosts) -> None:
        stats = tree.stats()
        costs.workers_input_tokens = stats["input_tokens"]
        costs.workers_output_tokens = stats["output_tokens"]
        costs.workers_cost = stats["workers_cost"]

    def _over_budget(self, tree: TaskTree, costs: SwarmCosts) -> bool:
        if self.max_cost is None:
            return False
        self._collect_worker_costs(tree, costs)
        if costs.total_cost >= self.max_cost:
            logger.warning(
                "budget exceeded: $%.4f >= $%.4f", costs.total_cost, self.max_cost
            )
            return True
        return False

    @staticmethod
    def _fallback_answer(tree: TaskTree) -> str | None:
        """Best-effort answer when the planner never called finish()."""
        root = tree.get(TaskTree.ROOT_ID)
        if root.result_summary:
            return root.result_summary
        summaries = tree.leaf_summaries()
        return "\n".join(summaries) if summaries else None
