"""Dynamic task tree shared between a planner and a fleet of workers.

The tree is the single source of truth of a swarm run. The planner mutates
it through its tools (decompose, retry, skip) and only ever *sees* it through
:meth:`TaskTree.render`, which exposes statuses and short summaries — never
raw worker outputs. Workers receive one leaf task each, plus the full outputs
of their direct dependencies only.

Built to scale to thousands of tasks:

- settled-state is tracked incrementally (per-parent counters), so
  completion checks and ``ready_tasks()`` scans are O(1) per node instead
  of recursive;
- ``render()`` collapses fully-settled subtrees to a single line and caps
  its output, so the planner's context stays bounded no matter how many
  tasks have run.

Concurrency: ``threading.Lock``, because agent frameworks commonly execute
tool functions in worker threads (e.g. ``asyncio.to_thread``) while the
scheduler reads the tree from the event loop.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any, Iterator


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


#: Statuses that satisfy a dependency and count toward parent completion.
_SETTLED = frozenset({TaskStatus.DONE, TaskStatus.SKIPPED})

#: Declared difficulty of a task. Deliberately three coarse buckets rather
#: than model names: the tree must not know which models a run uses, and a
#: planner asked to pick a model picks badly (it optimises for the name it
#: recognises). A worker_factory maps these onto a fleet.
EFFORTS = ("low", "standard", "high")


def _effort(value: Any) -> str:
    """Normalize a planner-declared effort; anything unknown is standard."""
    text = str(value or "").strip().lower()
    return text if text in EFFORTS else "standard"


@dataclass
class TaskNode:
    """One node of the task tree.

    ``result_summary`` is the 2-3 sentence digest shown to the planner;
    ``result_full`` is the raw worker output, only ever forwarded to
    dependent workers, never to the planner.
    """

    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    parent_id: str | None = None
    depends_on: list[str] = field(default_factory=list)
    #: How hard this task is, as declared by the planner: "low" | "standard"
    #: | "high". The tree only records it; a ``worker_factory`` is what turns
    #: it into a model choice (see ``adapters.*.build_swarm``).
    effort: str = "standard"
    children: list[str] = field(default_factory=list)
    result_summary: str | None = None
    result_full: str | None = None
    error: str | None = None
    worker: str | None = None
    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    duration: float = 0.0
    #: planner turn that created this task (0 = created outside a turn)
    turn: int = 0
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


#: Field names TaskNode accepts, used to filter a deserialized payload.
_NODE_FIELDS = frozenset(f.name for f in fields(TaskNode))


class TaskTreeError(ValueError):
    """Raised on invalid tree mutations (unknown ids, cycles, depth...).

    Messages are written for an LLM audience: planner tools catch this and
    return the text verbatim so the model can correct its call.
    """


class TaskTree:
    """Thread-safe dynamic task tree with incremental completion tracking."""

    ROOT_ID = "root"

    def __init__(self, goal: str, max_depth: int = 3, run_id: str | None = None):
        self.run_id = run_id or str(uuid.uuid4())
        self.max_depth = max_depth
        self._lock = threading.Lock()
        self._nodes: dict[str, TaskNode] = {
            self.ROOT_ID: TaskNode(id=self.ROOT_ID, description=goal)
        }
        #: settled children count per internal node (incremental bookkeeping)
        self._settled_children: dict[str, int] = {}
        #: one entry per planner deliberation: what it cost and what it planned
        self.planner_turns: list[dict[str, Any]] = []
        self._tasks_at_last_turn = 0
        self._turn = 0

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def goal(self) -> str:
        return self._nodes[self.ROOT_ID].description

    def get(self, node_id: str) -> TaskNode:
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                raise TaskTreeError(f"Unknown task id '{node_id}'.")
            return node

    def __len__(self) -> int:
        with self._lock:
            return len(self._nodes) - 1  # excluding root

    def is_complete(self) -> bool:
        with self._lock:
            return self._is_settled(self.ROOT_ID)

    def _is_settled(self, node_id: str) -> bool:
        """O(1) settled check thanks to incremental counters."""
        node = self._nodes[node_id]
        if node.is_leaf:
            return node.status in _SETTLED
        return self._settled_children.get(node_id, 0) == len(node.children)

    def _depth_unlocked(self, node_id: str) -> int:
        depth = 0
        node = self._nodes[node_id]
        while node.parent_id is not None:
            depth += 1
            node = self._nodes[node.parent_id]
        return depth

    # ------------------------------------------------------------------ #
    # Mutations
    # ------------------------------------------------------------------ #

    def decompose(
        self, parent_id: str, subtasks: list[dict[str, Any]]
    ) -> list[TaskNode]:
        """Add children to ``parent_id``. Atomic: all subtasks or none.

        Each subtask is ``{"description": str, "depends_on": [ref, ...]}``
        where a ref is either the 1-based index of another subtask in the
        same call, or the id of an existing sibling. Cycles among the new
        siblings are rejected. Calling decompose again on a node appends
        new siblings.
        """
        if not subtasks:
            raise TaskTreeError("decompose() called with an empty subtask list.")

        with self._lock:
            parent = self._nodes.get(parent_id)
            if parent is None:
                raise TaskTreeError(f"Unknown parent task id '{parent_id}'.")
            # A settled *container* is reopened by planning into it: that is
            # a second wave, which is how iterative planning works — the
            # planner reads the summaries of a finished batch and adds the
            # follow-ups it now knows it needs. Refusing that made the "plan
            # follow-up tasks" instruction impossible to obey.
            # A settled *leaf* is different: it has its own output, and
            # turning it into a container would orphan that.
            was_settled = self._is_settled(parent_id)
            if was_settled and parent.is_leaf:
                raise TaskTreeError(
                    f"Task '{parent_id}' is finished and has its own output; "
                    "retry it, or plan the follow-up as a sibling instead of "
                    "decomposing it."
                )
            depth = self._depth_unlocked(parent_id)
            if depth >= self.max_depth:
                raise TaskTreeError(
                    f"Max depth ({self.max_depth}) reached at task '{parent_id}'. "
                    "Make this task actionable instead of decomposing it."
                )

            offset = len(parent.children)
            prefix = "" if parent_id == self.ROOT_ID else f"{parent_id}."
            # (validation below may still reject the call; the reopen only
            # happens once the new children are actually attached)
            new_ids = [f"{prefix}{offset + i + 1}" for i in range(len(subtasks))]
            existing_siblings = set(parent.children)

            resolved_deps: list[list[str]] = []
            for i, sub in enumerate(subtasks):
                description = str(sub.get("description", "")).strip()
                if not description:
                    raise TaskTreeError(f"Subtask #{i + 1} has an empty description.")
                deps: list[str] = []
                for ref in sub.get("depends_on") or []:
                    if isinstance(ref, int) or (
                        isinstance(ref, str)
                        and ref.isdigit()
                        and f"{prefix}{ref}" not in existing_siblings
                    ):
                        idx = int(ref)
                        if not 1 <= idx <= len(subtasks):
                            raise TaskTreeError(
                                f"Subtask #{i + 1} depends on index {idx}, but this "
                                f"call only defines {len(subtasks)} subtasks."
                            )
                        if idx == i + 1:
                            raise TaskTreeError(f"Subtask #{i + 1} depends on itself.")
                        deps.append(new_ids[idx - 1])
                    else:
                        ref = str(ref)
                        if ref not in existing_siblings and ref not in new_ids:
                            raise TaskTreeError(
                                f"Subtask #{i + 1} depends on '{ref}', which is not "
                                f"a sibling under '{parent_id}'. Dependencies may "
                                "only reference sibling tasks."
                            )
                        deps.append(ref)
                resolved_deps.append(deps)

            self._check_acyclic(new_ids, resolved_deps)

            # If the parent was a leaf about to become internal, it must not
            # carry settled-leaf state.
            created: list[TaskNode] = []
            for node_id, sub, deps in zip(new_ids, subtasks, resolved_deps):
                node = TaskNode(
                    id=node_id,
                    description=str(sub["description"]).strip(),
                    parent_id=parent_id,
                    depends_on=deps,
                    effort=_effort(sub.get("effort")),
                    turn=self._turn,
                )
                self._nodes[node_id] = node
                parent.children.append(node_id)
                created.append(node)

            parent.status = TaskStatus.IN_PROGRESS
            self._settled_children.setdefault(parent_id, 0)
            if was_settled:
                # It had auto-completed: drop the summary it inherited from
                # its children and walk the ancestors' counters back, exactly
                # as reopening a leaf does.
                parent.completed_at = None
                parent.result_summary = None
                parent.result_full = None
                self._on_unsettled(parent.parent_id)
            return created

    @staticmethod
    def _check_acyclic(ids: list[str], deps: list[list[str]]) -> None:
        graph = {i: [d for d in dd if d in ids] for i, dd in zip(ids, deps)}
        state: dict[str, int] = {}  # 0=unseen 1=visiting 2=done

        def visit(n: str) -> None:
            if state.get(n) == 1:
                raise TaskTreeError(f"Dependency cycle detected involving task '{n}'.")
            if state.get(n) == 2:
                return
            state[n] = 1
            for m in graph.get(n, []):
                visit(m)
            state[n] = 2

        for n in ids:
            visit(n)

    def mark_in_progress(self, node_id: str, worker: str | None = None) -> None:
        with self._lock:
            node = self._nodes[node_id]
            node.status = TaskStatus.IN_PROGRESS
            node.worker = worker
            node.attempts += 1

    def mark_done(
        self,
        node_id: str,
        summary: str,
        full: str = "",
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
        duration: float = 0.0,
    ) -> None:
        with self._lock:
            node = self._nodes[node_id]
            was_settled = node.status in _SETTLED
            node.status = TaskStatus.DONE
            node.result_summary = summary.strip()
            node.result_full = full or summary
            node.error = None
            node.input_tokens += input_tokens
            node.output_tokens += output_tokens
            node.cost += cost
            node.duration += duration
            node.completed_at = time.time()
            if not was_settled:
                self._on_settled(node.parent_id)

    def _on_settled(self, parent_id: str | None) -> None:
        """A child just settled: bump counters, auto-complete full parents."""
        while parent_id is not None:
            self._settled_children[parent_id] = (
                self._settled_children.get(parent_id, 0) + 1
            )
            parent = self._nodes[parent_id]
            if self._settled_children[parent_id] < len(parent.children):
                return
            # Every child settled -> the parent auto-completes.
            parent.status = TaskStatus.DONE
            parent.completed_at = time.time()
            if parent.result_summary is None:
                parts = [
                    self._nodes[c].result_summary
                    for c in parent.children
                    if self._nodes[c].result_summary
                ]
                parent.result_summary = " ".join(parts)[:2000] or None
            parent_id = parent.parent_id

    def mark_failed(
        self,
        node_id: str,
        error: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
    ) -> None:
        """Record a failed attempt — including what it spent failing.

        A worker that burns its whole output budget and returns nothing has
        cost real money. Booking only successes made those tokens invisible
        to ``stats()`` and therefore to ``max_cost``, so a run failing
        expensively looked free right up to the provider's invoice.
        """
        with self._lock:
            node = self._nodes[node_id]
            node.status = TaskStatus.FAILED
            node.error = error.strip()[:2000]
            node.input_tokens += input_tokens
            node.output_tokens += output_tokens
            node.cost += cost

    def reset_task(self, node_id: str) -> None:
        """Send a failed (or settled) leaf back to PENDING for a fresh attempt.

        Un-settles auto-completed ancestors so the branch reopens cleanly.
        """
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                raise TaskTreeError(f"Unknown task id '{node_id}'.")
            if not node.is_leaf:
                raise TaskTreeError(
                    f"Task '{node_id}' has subtasks; retry one of its leaves instead."
                )
            was_settled = node.status in _SETTLED
            node.status = TaskStatus.PENDING
            node.error = None
            node.result_summary = None
            node.result_full = None
            node.worker = None
            if was_settled:
                self._on_unsettled(node.parent_id)

    def _on_unsettled(self, parent_id: str | None) -> None:
        """Reverse of :meth:`_on_settled` when a settled leaf is reset."""
        while parent_id is not None:
            parent = self._nodes[parent_id]
            was_full = (
                self._settled_children.get(parent_id, 0) == len(parent.children)
            )
            self._settled_children[parent_id] = max(
                0, self._settled_children.get(parent_id, 0) - 1
            )
            if not was_full:
                return
            # The parent had auto-completed; reopen it.
            parent.status = TaskStatus.IN_PROGRESS
            parent.completed_at = None
            parent.result_summary = None
            parent.result_full = None
            parent_id = parent.parent_id

    def reopen_running(self) -> list[str]:
        """Send every IN_PROGRESS leaf back to PENDING; returns their ids.

        A tree loaded from disk was written by a run that was killed, so its
        "running" tasks have no worker behind them. Nothing would ever settle
        them and ``is_complete()`` would never be true — the resumed run
        would hang on tasks that died with the previous process.
        """
        with self._lock:
            reopened = []
            for node in self._nodes.values():
                if (
                    node.id != self.ROOT_ID
                    and node.is_leaf
                    and node.status is TaskStatus.IN_PROGRESS
                ):
                    node.status = TaskStatus.PENDING
                    node.worker = None
                    reopened.append(node.id)
            return reopened

    def skip_task(self, node_id: str, reason: str) -> None:
        """Mark a leaf as intentionally abandoned; unblocks its dependents."""
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                raise TaskTreeError(f"Unknown task id '{node_id}'.")
            if not node.is_leaf:
                raise TaskTreeError(
                    f"Task '{node_id}' has subtasks and cannot be skipped."
                )
            was_settled = node.status in _SETTLED
            node.status = TaskStatus.SKIPPED
            node.result_summary = f"[skipped] {reason.strip()}"
            node.result_full = node.result_summary
            node.completed_at = time.time()
            if not was_settled:
                self._on_settled(node.parent_id)

    # ------------------------------------------------------------------ #
    # Scheduling queries
    # ------------------------------------------------------------------ #

    def ready_tasks(self, limit: int | None = None) -> list[TaskNode]:
        """Pending leaves whose dependencies are all settled, hottest first.

        Ordered by **critical path**: a task with a long chain of tasks
        waiting behind it goes out before an independent one, because every
        second it waits is a second added to the run's total wall clock. Ties
        break on id, so a flat fan-out (no dependency at all, the common
        case) keeps its natural order and pays nothing for the ordering.

        O(N) scan with O(1) checks per node; pass ``limit`` to stop early
        on very large trees when only a dispatch batch is needed.
        """
        with self._lock:
            ready: list[TaskNode] = []
            for node in self._nodes.values():
                if node.id == self.ROOT_ID or not node.is_leaf:
                    continue
                if node.status is not TaskStatus.PENDING:
                    continue
                if all(
                    self._is_settled(dep)
                    for dep in node.depends_on
                    if dep in self._nodes
                ):
                    ready.append(node)
            chains = self._chain_lengths_unlocked()
            if chains:
                ready.sort(key=lambda n: (-self._priority(n, chains), _id_sort_key(n.id)))
            else:
                ready.sort(key=lambda n: _id_sort_key(n.id))
            return ready[:limit] if limit else ready

    def _priority(self, node: TaskNode, chains: dict[str, int]) -> int:
        """Longest chain waiting on this leaf, directly or via an ancestor.

        A leaf under a container is on the hook for that container's own
        dependents too: nothing depending on "section 3" starts until every
        leaf of section 3 is done. Counting only the leaf's own dependents
        would rank those leaves as if nothing waited on them.
        """
        best = chains.get(node.id, 0)
        parent_id = node.parent_id
        while parent_id is not None:
            best = max(best, chains.get(parent_id, 0))
            parent_id = self._nodes[parent_id].parent_id
        return best

    def _chain_lengths_unlocked(self) -> dict[str, int]:
        """Longest chain of tasks downstream of each node, or ``{}`` if flat.

        Iterative rather than recursive: a thousand-task pipeline would blow
        the stack, and this runs on every dispatch. Returns empty when no
        task declares a dependency, which lets ``ready_tasks`` skip sorting
        work entirely on a pure fan-out.
        """
        dependents: dict[str, list[str]] = {}
        for node in self._nodes.values():
            for dep in node.depends_on:
                dependents.setdefault(dep, []).append(node.id)
        if not dependents:
            return {}

        # Kahn over the dependents graph, then fold in reverse: every node is
        # seen after all of its dependents, so one pass is enough. A node
        # caught in a cycle never enters `order` and stays absent from
        # `chains`, which reads as priority 0 — degraded ordering, never a
        # hang.
        indegree = {nid: 0 for nid in self._nodes}
        for children in dependents.values():
            for dependent in children:
                if dependent in indegree:
                    indegree[dependent] += 1
        queue = deque(nid for nid, deg in indegree.items() if deg == 0)
        order: list[str] = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for dependent in dependents.get(nid, ()):
                if dependent not in indegree:
                    continue
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    queue.append(dependent)

        chains: dict[str, int] = {}
        for nid in reversed(order):
            chains[nid] = max(
                (1 + chains.get(d, 0) for d in dependents.get(nid, ())), default=0
            )
        return chains

    def failed_tasks(self) -> list[TaskNode]:
        with self._lock:
            return [n for n in self._nodes.values() if n.status is TaskStatus.FAILED]

    def dependency_blocks(self, node_id: str) -> list[dict[str, str]]:
        """One block per direct dependency of ``node_id``, in declared order.

        Each block is ``{"id", "description", "summary", "full"}``. For a
        dependency that was itself decomposed, the outputs of its leaf
        descendants are concatenated.

        The summary travels with the full text because whoever assembles the
        worker's context may have to cut it (see
        ``worker.build_worker_input``): a trimmed dependency is far more
        useful introduced by its own digest than by a sentence sliced off
        mid-word.
        """
        with self._lock:
            node = self._nodes[node_id]
            blocks: list[dict[str, str]] = []
            for dep_id in node.depends_on:
                dep = self._nodes.get(dep_id)
                if dep is None:
                    continue
                chunks = [
                    leaf.result_full
                    for leaf in self._iter_leaves_unlocked(dep_id)
                    if leaf.result_full
                ]
                if not chunks:
                    continue
                blocks.append(
                    {
                        "id": dep.id,
                        "description": dep.description,
                        "summary": dep.result_summary or "",
                        "full": "\n\n".join(chunks),
                    }
                )
            return blocks

    def dependency_context(self, node_id: str) -> dict[str, str]:
        """Full outputs of the direct dependencies, keyed ``"<id> — <desc>"``."""
        return {
            f"{b['id']} — {b['description']}": b["full"]
            for b in self.dependency_blocks(node_id)
        }

    def _iter_leaves_unlocked(self, node_id: str) -> Iterator[TaskNode]:
        node = self._nodes[node_id]
        if node.is_leaf:
            yield node
            return
        for child in node.children:
            yield from self._iter_leaves_unlocked(child)

    def leaf_summaries(self) -> list[str]:
        with self._lock:
            return [
                f"[{leaf.id}] {leaf.result_summary}"
                for leaf in self._iter_leaves_unlocked(self.ROOT_ID)
                if leaf.result_summary
            ]

    # ------------------------------------------------------------------ #
    # Rendering & stats
    # ------------------------------------------------------------------ #

    _GLYPHS = {
        TaskStatus.PENDING: "·",
        TaskStatus.IN_PROGRESS: "▶",
        TaskStatus.DONE: "✔",
        TaskStatus.FAILED: "✖",
        TaskStatus.SKIPPED: "⤼",
    }

    def render(
        self,
        max_summary_chars: int = 280,
        collapse_done: bool = True,
        max_lines: int = 250,
    ) -> str:
        """Compact textual view of the tree — the planner's only window.

        Bounded even on thousand-task runs: fully-settled subtrees collapse
        to a single line (``collapse_done``) and the output is capped at
        ``max_lines`` (middle truncation), followed by a stats footer.
        Raw outputs are deliberately absent.
        """
        with self._lock:
            lines = [f"GOAL: {self.goal}"]
            root = self._nodes[self.ROOT_ID]
            if root.is_leaf:
                lines.append("(no tasks planned yet)")
            else:
                for child in root.children:
                    self._render_node(child, lines, 0, max_summary_chars, collapse_done)
            if len(lines) > max_lines:
                keep_head = max_lines * 2 // 3
                keep_tail = max_lines - keep_head
                omitted = len(lines) - keep_head - keep_tail
                lines = (
                    lines[:keep_head]
                    + [f"… (+{omitted} lines omitted — settled tasks mostly)"]
                    + lines[-keep_tail:]
                )
            lines.append(self._footer())
            return "\n".join(lines)

    def _render_node(
        self,
        node_id: str,
        lines: list[str],
        indent: int,
        max_chars: int,
        collapse_done: bool,
    ) -> None:
        node = self._nodes[node_id]
        pad = "  " * indent
        glyph = self._GLYPHS[node.status]
        deps = f" (after {', '.join(node.depends_on)})" if node.depends_on else ""
        retries = f" [attempt {node.attempts}]" if node.attempts > 1 else ""
        # Only a non-default effort shows: the planner needs to see that its
        # declaration landed, without paying a token on the 90% of tasks
        # that are standard.
        effort = f" <{node.effort}>" if node.effort != "standard" else ""

        if collapse_done and not node.is_leaf and self._is_settled(node_id):
            count = sum(1 for _ in self._iter_leaves_unlocked(node_id))
            summary = f" — {node.result_summary[:max_chars]}" if node.result_summary else ""
            lines.append(f"{pad}✔ [{node.id}] {node.description} ({count} tasks done){summary}")
            return

        lines.append(f"{pad}{glyph} [{node.id}] {node.description}{effort}{deps}{retries}")
        if node.status is TaskStatus.FAILED and node.error:
            lines.append(f"{pad}    error: {node.error[:max_chars]}")
        elif node.result_summary and node.is_leaf:
            lines.append(f"{pad}    → {node.result_summary[:max_chars]}")
        for child in node.children:
            self._render_node(child, lines, indent + 1, max_chars, collapse_done)

    def _footer(self) -> str:
        leaves = [n for n in self._nodes.values() if n.is_leaf and n.id != self.ROOT_ID]
        by_status: dict[str, int] = {}
        for n in leaves:
            by_status[n.status.value] = by_status.get(n.status.value, 0) + 1
        cost = sum(n.cost for n in leaves)
        parts = ", ".join(f"{v} {k}" for k, v in sorted(by_status.items()))
        return f"[{len(leaves)} tasks: {parts or 'none'} — workers cost ${cost:.4f}]"

    def mark_planner_thinking(self, worker: str) -> None:
        """Flag the root as a planner turn starts.

        Two things only this moment knows: the planner is *working* (so the
        root should read as running, not idle), and which model it is. Both
        used to appear only once the turn returned — minutes later on a wide
        plan, which is exactly when a user is staring at the graph wondering
        whether anything is happening. Token counts still cannot: the
        provider reports usage when the call ends.
        """
        with self._lock:
            root = self._nodes[self.ROOT_ID]
            root.worker = worker
            root.status = TaskStatus.IN_PROGRESS
            self._turn += 1
            self.planner_turns.append({
                "n": self._turn, "worker": worker, "running": True,
                "itok": 0, "otok": 0, "cost": 0.0, "dur": 0.0, "tasks": 0,
            })

    def record_planner_turn(
        self,
        worker: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        seconds: float,
    ) -> dict[str, Any]:
        """Book a planner turn against the root node.

        The planner is an agent like any other and its spend is part of the
        run — but it has no leaf of its own, so its tokens used to be
        invisible everywhere the tree is the source of truth (the live cost
        line, the graph, the node panel). The root carries them: cumulative,
        one "attempt" per turn.
        """
        with self._lock:
            root = self._nodes[self.ROOT_ID]
            root.worker = worker
            root.attempts += 1
            prev_in, prev_out, prev_cost = (
                root.input_tokens, root.output_tokens, root.cost
            )
            root.input_tokens = input_tokens
            root.output_tokens = output_tokens
            root.cost = cost
            root.duration += seconds
            root.status = TaskStatus.PENDING
            # Per-turn deltas: what *this* deliberation cost and planned.
            # The cumulative figures live on the root; a reader wanting the
            # sequence ("it thought again, and added three tasks") needs the
            # difference, which only this method can compute.
            turn = self.planner_turns[-1] if self.planner_turns else {}
            turn.update({
                "n": self._turn,
                "worker": worker,
                "running": False,
                "itok": input_tokens - prev_in,
                "otok": output_tokens - prev_out,
                "cost": round(cost - prev_cost, 6),
                "dur": round(seconds, 2),
                "tasks": len(self._nodes) - 1 - self._tasks_at_last_turn,
            })
            self._tasks_at_last_turn = len(self._nodes) - 1
            return turn

    def stats(self) -> dict[str, Any]:
        with self._lock:
            leaves = [
                n for n in self._nodes.values() if n.is_leaf and n.id != self.ROOT_ID
            ]
            by_status: dict[str, int] = {}
            for n in leaves:
                by_status[n.status.value] = by_status.get(n.status.value, 0) + 1
            return {
                "tasks": len(leaves),
                "by_status": by_status,
                "input_tokens": sum(n.input_tokens for n in leaves),
                "output_tokens": sum(n.output_tokens for n in leaves),
                # Leaves are the workers, the root carries the planner.
                # "cost" is the headline total (that is what a status line or
                # a header should show); the split is there for accounting.
                "cost": sum(n.cost for n in leaves)
                        + self._nodes[self.ROOT_ID].cost,
                "workers_cost": sum(n.cost for n in leaves),
                "planner_cost": self._nodes[self.ROOT_ID].cost,
            }

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "run_id": self.run_id,
                "max_depth": self.max_depth,
                "turn": self._turn,
                "planner_turns": list(self.planner_turns),
                "nodes": {nid: n.to_dict() for nid, n in self._nodes.items()},
            }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskTree":
        tree = cls.__new__(cls)
        tree.run_id = data["run_id"]
        tree.max_depth = data.get("max_depth", 3)
        tree._lock = threading.Lock()
        tree._nodes = {}
        tree._settled_children = {}
        # Turn bookkeeping must be restored, not just the nodes: a resumed
        # run calls mark_planner_thinking on its first turn, and these are
        # what it increments. They were absent until a run was resumed for
        # the first time, which is how an AttributeError hid here.
        tree._turn = int(data.get("turn", 0))
        tree.planner_turns = list(data.get("planner_turns", []))
        for nid, nd in data["nodes"].items():
            nd = dict(nd)
            nd["status"] = TaskStatus(nd["status"])
            # Forward compatibility: a tree written by an older version has
            # no `effort`, a newer one may have fields this build ignores.
            nd = {k: v for k, v in nd.items() if k in _NODE_FIELDS}
            tree._nodes[nid] = TaskNode(**nd)
        tree._tasks_at_last_turn = len(tree._nodes) - 1
        # Rebuild the incremental counters bottom-up.
        for nid, node in tree._nodes.items():
            if not node.is_leaf:
                tree._settled_children[nid] = sum(
                    1
                    for c in node.children
                    if tree._nodes[c].status in _SETTLED
                    or (
                        not tree._nodes[c].is_leaf
                        and tree._nodes[c].status is TaskStatus.DONE
                    )
                )
        return tree


def _id_sort_key(node_id: str) -> tuple[int, ...]:
    try:
        return tuple(int(p) for p in node_id.split("."))
    except ValueError:
        return (1 << 30,)
