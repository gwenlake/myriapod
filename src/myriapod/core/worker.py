"""Worker side of the swarm: run one leaf task with minimal context.

Framework-neutral: drives any :class:`myriapod.core.protocol.AgentLike`.
A worker receives its task description plus the full outputs of its direct
dependencies, and must end its answer with a ``<summary>`` block — the only
part the planner will ever see.

Retries of the same task back off exponentially (with jitter) *inside* the
worker coroutine, so a rate-limited fleet self-paces without any scheduler
complexity — essential when hundreds of workers hit the same provider.

Dependency output is fitted to a character budget rather than concatenated
whole, and an optional :class:`ReviewPolicy` puts a judge between the answer
and the tree (see ``core.reviewer``).
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from myriapod.core.protocol import AgentLike
from myriapod.core.reviewer import review_answer, revision_note
from myriapod.core.task_tree import TaskNode, TaskTree

logger = logging.getLogger("myriapod")

WORKER_INSTRUCTIONS = """\
You are a WORKER agent in a swarm. You have exactly one task, given below.
Complete it alone: no follow-up questions, no delegation, no request for
clarification. Where the brief is ambiguous, take the most useful reading,
proceed, and state which reading you took.

Depth is what you are here for. The brief says what to cover; cover it
properly:

- Be specific. Name mechanisms, actors, dates, quantities with units. A
  sentence a reader could have written without your task is filler.
- Give the reasoning behind a non-obvious claim, not the claim alone.
- Where the evidence is contested or you are unsure, say so and say why. A
  stated uncertainty is worth more than hedged prose that never commits.
- No preamble, no restating the task, no "in conclusion" padding, and no
  placeholders like TBD. Fill the space with content or stop.

If you have search or retrieval tools, use them before answering: start
broad, see what exists, then narrow. Stop once further queries return what
you already have.

Write the deliverable in Markdown — ## headings, bullets, tables, fenced
code where they earn their place. It may be assembled into a document or a
slide deck.

End your answer with:
<summary>2-4 sentences: what you produced, the key findings or figures, and
any gap or caveat the coordinator should know about.</summary>

The coordinator reads only the summary and routes on it, so make it
informative rather than procedural — "Three fragments survive, dated to
roughly 100 BCE..." beats "I completed the task". Everything above the
summary is what dependent workers receive."""

_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL | re.IGNORECASE)

#: Prefixed to the summary when the worker never closed a ``<summary>`` block.
#: The overwhelmingly common cause is the answer being cut off at the model's
#: output limit, which otherwise reaches the planner looking perfectly healthy.
TRUNCATED_MARKER = "[no <summary> block — the answer is probably cut off]"

#: Base delay (seconds) for retry backoff; attempt n waits base * 2^(n-2).
RETRY_BACKOFF_BASE = 2.0
RETRY_BACKOFF_MAX = 30.0

#: How much dependency output one worker may receive, in characters
#: (~4 chars/token, so ~30k tokens). Chosen to leave a wide margin under a
#: 200k-token window once the system prompt, the brief and the answer are
#: counted — the ceiling that matters in practice is not the context window
#: but attention: a model handed 80k tokens of other workers' prose reads
#: the first and last few thousand and averages the rest.
DEFAULT_CONTEXT_CHARS = 120_000

#: Below this many characters, an excerpt of a dependency is not worth
#: including: what fits is a heading and an intro, which reads as the whole
#: answer and is not one. Those dependencies are passed as their digest.
MIN_EXCERPT_CHARS = 3_000

#: Sentinel: "no pricing given", distinct from "explicitly unpriced (None)".
_UNSET = object()


def build_worker_input(
    tree: TaskTree,
    node: TaskNode,
    context_chars: int = DEFAULT_CONTEXT_CHARS,
) -> tuple[str, str | None]:
    """(task message, context string) for one leaf task, within a budget.

    A synthesis task that declares twenty dependencies used to receive the
    concatenation of twenty full worker answers — tens of thousands of
    tokens, growing with the fan-out, until the call was rejected or the
    model quietly lost the middle of it. The dependencies now share
    ``context_chars`` by **fair share with spillover**: every dependency is
    granted an equal slice, whatever a small one leaves unused is
    redistributed to the large ones, and only then is anything trimmed. A
    trimmed dependency keeps its head and is introduced by its own summary,
    so the worker always knows what it is missing rather than reading a
    sentence that stops mid-word; one trimmed past
    :data:`MIN_EXCERPT_CHARS` is passed as its digest alone (see
    :func:`_allocate`).

    The digests themselves are never dropped: they are the floor of what a
    dependent worker must know, and they cost a couple of hundred characters
    each. A budget too small to hold them yields all-digests, not silence.
    """
    message = f"TASK [{node.id}]: {node.description}"
    blocks = tree.dependency_blocks(node.id)
    if not blocks:
        return message, None

    budgets = _allocate(blocks, context_chars)
    rendered = [
        _render_dependency(block, budget) for block, budget in zip(blocks, budgets)
    ]
    return message, "\n\n".join(rendered)


def _allocate(blocks: list[dict[str, str]], budget: int) -> list[int]:
    """Per-dependency character grants; ``0`` means "digest only".

    Fair share alone breaks down on a wide fan-in: forty dependencies
    sharing 50k characters get 1250 each, and 1250 characters of a report is
    its introduction — a worker reading forty introductions is worse
    informed than one reading forty digests, and believes itself better
    informed. So any dependency whose share falls under
    :data:`MIN_EXCERPT_CHARS` is dropped to its digest, which frees its
    share for the dependencies that can still be quoted usefully. Dropping
    one may lift another over the floor, hence the loop.
    """
    sizes = [len(b["full"]) for b in blocks]
    digest_only: set[int] = set()
    while True:
        pool = [i for i in range(len(blocks)) if i not in digest_only]
        if not pool:
            return [0] * len(blocks)
        spent = sum(len(blocks[i]["summary"]) for i in digest_only)
        grants = _fair_share([sizes[i] for i in pool], max(budget - spent, 0))
        demoted = {
            i
            for i, grant in zip(pool, grants)
            if grant < sizes[i] and grant < MIN_EXCERPT_CHARS and blocks[i]["summary"]
        }
        if not demoted:
            allocation = [0] * len(blocks)
            for i, grant in zip(pool, grants):
                allocation[i] = grant
            return allocation
        digest_only |= demoted


def _fair_share(sizes: list[int], budget: int) -> list[int]:
    """Water-filling: equal slices, with what the small ones don't use.

    Repeatedly hands every still-hungry item an equal share of what is left,
    so one 40k-char dependency never starves nine 2k ones — and nine small
    ones never waste 90% of the budget on padding they don't need.
    """
    granted = [0] * len(sizes)
    remaining = max(budget, 0)
    hungry = [i for i, size in enumerate(sizes) if size > 0]
    while hungry and remaining >= len(hungry):
        share = remaining // len(hungry)
        still_hungry = []
        for i in hungry:
            take = min(sizes[i] - granted[i], share)
            granted[i] += take
            remaining -= take
            if granted[i] < sizes[i]:
                still_hungry.append(i)
        # If nobody was satiated this round, everyone took the full share and
        # `remaining` just dropped below one share each — the guard above
        # ends the loop. Either way this terminates.
        hungry = still_hungry
    return granted


def _render_dependency(block: dict[str, str], budget: int) -> str:
    """One dependency as the worker sees it: whole, excerpted, or digested."""
    header = f"### Output of dependency {block['id']} — {block['description']}"
    full = block["full"]
    if len(full) <= budget:
        return f"{header}\n{full}"
    if budget <= 0:
        return (
            f"{header} (digest only — the full output is {len(full)} characters, "
            "too long to include alongside the other dependencies)\n"
            f"{block['summary']}"
        )
    cut = full[:budget].rstrip()
    dropped = len(full) - len(cut)
    digest = (
        f"\n_Digest of the full output: {block['summary']}_" if block["summary"] else ""
    )
    return (
        f"{header} (excerpt: first {len(cut)} of {len(full)} characters)"
        f"{digest}\n{cut}\n\n[… {dropped} characters truncated to fit the "
        "context budget. Work from what is above and from the digest; do not "
        "invent the missing part — say so if it matters.]"
    )


def split_summary(content: str) -> tuple[str, str]:
    """(summary, full_without_summary_tag) from a worker answer."""
    content = (content or "").strip()
    match = _SUMMARY_RE.search(content)
    if match:
        summary = " ".join(match.group(1).split()).strip()
        full = _SUMMARY_RE.sub("", content).strip()
        return summary or full[:300], full or summary
    return content[:300], content


def compute_cost(
    input_tokens: int, output_tokens: int, pricing: dict[str, float] | None
) -> float:
    """Dollar cost from token counts and a per-million-token pricing table."""
    if not pricing:
        return 0.0
    return (
        input_tokens * pricing.get("input", 0.0)
        + output_tokens * pricing.get("output", 0.0)
    ) / 1_000_000


def resolve_pricing(pricing: dict[str, Any] | None, effort: str) -> dict | None:
    """The rate that applies to one task: flat table, or one per effort tier.

    A fleet routing "high" tasks to a stronger model bills them at that
    model's rate — ``{"low": {...}, "high": {...}}``. Without this, a run
    that routes half its tasks to a model three times the price reports the
    cheap price for all of them, and ``max_cost`` stops protecting anything.
    """
    if not pricing:
        return None
    if any(isinstance(value, dict) for value in pricing.values()):
        return pricing.get(effort) or pricing.get("standard") or None
    return pricing


@dataclass(slots=True)
class ReviewPolicy:
    """How (and whether) a worker's answer is judged before it is accepted.

    ``reviewer_factory`` builds the judge for a task — cheap model, no tools.
    ``worker_factory`` rebuilds the worker for a revision: agents in some
    frameworks carry conversation history, and a revision must not inherit
    the answer it is replacing. Without it, the original agent is reused.
    """

    reviewer_factory: Callable[[TaskNode], AgentLike]
    worker_factory: Callable[[TaskNode], AgentLike] | None = None
    pricing: dict[str, float] | None = None
    max_revisions: int = 1


async def run_worker(
    agent: AgentLike,
    tree: TaskTree,
    node: TaskNode,
    timeout: float | None = None,
    pricing: dict[str, float] | None = None,
    context_chars: int = DEFAULT_CONTEXT_CHARS,
    review: ReviewPolicy | None = None,
) -> None:
    """Execute one leaf task and record the outcome in the tree.

    Never raises (except cancellation): failures become FAILED tasks so the
    scheduler and the planner can react (auto-retry, retry_task, skip_task).

    With a :class:`ReviewPolicy`, the answer is judged before it lands, and a
    rejected answer is rewritten by a fresh worker holding the review. Tokens
    from every attempt and every review are booked on the task, so the run's
    cost is what it actually spent, not what the last attempt cost.
    """
    name = getattr(agent, "name", None) or type(agent).__name__
    tree.mark_in_progress(node.id, worker=name)

    # Backoff on retries: attempt 2 waits ~2s, attempt 3 ~4s... capped, jittered.
    if node.attempts > 1:
        delay = min(
            RETRY_BACKOFF_BASE * 2 ** (node.attempts - 2), RETRY_BACKOFF_MAX
        ) * random.uniform(0.5, 1.5)
        await asyncio.sleep(delay)

    message, context = build_worker_input(tree, node, context_chars)
    started = time.time()
    spend = _Spend(pricing)

    outcome = await _ask(agent, message, context, timeout)
    if isinstance(outcome, _Failure):
        tree.mark_failed(node.id, outcome.error)
        if outcome.cancelled:
            raise asyncio.CancelledError
        return
    spend.add(outcome)

    content = (getattr(outcome, "content", None) or "").strip()
    if not content:
        # Say what it spent producing nothing: that number is the whole
        # diagnosis. Near zero means the call never really ran (auth, an
        # empty prompt). Thousands of tokens means the agent burned its
        # budget on tool calls or a truncated reply and never wrote a final
        # answer — which on this fleet is a wide fan-in, and is fixed by
        # splitting the task, not by retrying it.
        #
        # An adapter that can tell the causes apart overrides that guess
        # (``empty_reason``): "the answer never fit" and "the model never got
        # to answer" both arrive here as an empty string and imply opposite
        # fixes, and only the adapter knows the model's output ceiling.
        reason = getattr(outcome, "empty_reason", None) or (
            "If that count is high, the task is too big for one answer: "
            "split it."
        )
        tree.mark_failed(
            node.id,
            "Worker returned an empty answer after "
            f"{spend.output_tokens} output tokens (no final text). {reason}",
            input_tokens=spend.input_tokens,
            output_tokens=spend.output_tokens,
            cost=spend.cost,
        )
        return
    # The call that produced the answer we keep — not the task's total spend.
    # A truncation is diagnosed by comparing this against the model's output
    # ceiling, and the cumulative figure (attempts + reviews) reads as far
    # over any ceiling, which points at the wrong setting.
    answer_tokens = getattr(outcome, "output_tokens", 0)

    if review is not None:
        content, answer_tokens = await _revise_until_accepted(
            content, answer_tokens, message, context, node, timeout, review, spend
        )

    summary, full = split_summary(content)
    if not _SUMMARY_RE.search(content):
        logger.warning(
            "task %s returned no <summary> block (%d output tokens in the "
            "answering call) — likely truncated",
            node.id,
            answer_tokens,
        )
        summary = f"{TRUNCATED_MARKER} {summary}"
    tree.mark_done(
        node.id,
        summary,
        full,
        input_tokens=spend.input_tokens,
        output_tokens=spend.output_tokens,
        cost=spend.cost,
        duration=time.time() - started,
    )


@dataclass(slots=True)
class _Failure:
    """An agent call that produced no answer."""

    error: str
    cancelled: bool = False


class _Spend:
    """Running total of what one task cost across attempts and reviews.

    A reviewed task bills several model calls, possibly on different models.
    Booking only the last one would make a swarm that revises look cheaper
    than one that does not — the exact figure a cost cap is enforced on.
    """

    __slots__ = ("input_tokens", "output_tokens", "cost", "worker_pricing")

    def __init__(self, worker_pricing: dict[str, float] | None) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0
        self.worker_pricing = worker_pricing

    def add(self, outcome: Any, pricing: Any = _UNSET) -> None:
        """Book one call, at worker pricing unless another rate is given."""
        itok = getattr(outcome, "input_tokens", 0)
        otok = getattr(outcome, "output_tokens", 0)
        self.input_tokens += itok
        self.output_tokens += otok
        # Priced at the rate of whoever ran: a Haiku review inside a Sonnet
        # task must not be billed as Sonnet, and vice versa.
        if pricing is _UNSET:
            pricing = self.worker_pricing
        self.cost += compute_cost(itok, otok, pricing)


async def _ask(
    agent: AgentLike, message: str, context: str | None, timeout: float | None
) -> Any:
    """One agent call: the outcome, or a :class:`_Failure` to record."""
    try:
        coro = agent.run(message, context=context)
        return await (asyncio.wait_for(coro, timeout) if timeout else coro)
    except asyncio.TimeoutError:
        return _Failure(f"Worker timed out after {timeout:.0f}s.")
    except asyncio.CancelledError:
        return _Failure("Worker cancelled (swarm shutdown).", cancelled=True)
    except Exception as e:  # noqa: BLE001 — any worker crash becomes a FAILED task
        return _Failure(f"{type(e).__name__}: {e}")


async def _revise_until_accepted(
    content: str,
    answer_tokens: int,
    message: str,
    context: str | None,
    node: TaskNode,
    timeout: float | None,
    review: ReviewPolicy,
    spend: _Spend,
) -> tuple[str, int]:
    """Judge, and rewrite while rejected. Returns (answer to keep, its tokens).

    A revision that fails or comes back empty leaves the previous answer in
    place: the gate may only ever improve a task's output, never cost it one.
    """
    # The judge is shown the brief, never the dependency payloads: it is
    # judging whether the answer delivers what was asked, and twenty
    # dependency outputs would cost more than the task itself to review.
    brief = message
    if context:
        brief += (
            "\n\n(The worker was also given the full outputs of its declared "
            "dependencies, which are not reproduced here.)"
        )
    for round_no in range(1, review.max_revisions + 1):
        verdict = await review_answer(
            review.reviewer_factory(node), brief, content, timeout
        )
        spend.add(verdict, review.pricing)
        if verdict.passed:
            return content, answer_tokens
        logger.info(
            "task %s rejected by review (round %d): %s",
            node.id, round_no, verdict.feedback[:200],
        )
        agent = review.worker_factory(node) if review.worker_factory else None
        if agent is None:
            return content, answer_tokens
        revised_context = "\n\n".join(
            part for part in (context, revision_note(verdict.feedback, content)) if part
        )
        outcome = await _ask(agent, message, revised_context, timeout)
        if isinstance(outcome, _Failure):
            if outcome.cancelled:
                raise asyncio.CancelledError
            logger.warning(
                "revision of task %s failed (%s); keeping the original",
                node.id, outcome.error,
            )
            return content, answer_tokens
        spend.add(outcome)
        revised = (getattr(outcome, "content", None) or "").strip()
        if not revised:
            return content, answer_tokens
        content = revised
        answer_tokens = getattr(outcome, "output_tokens", 0)
    return content, answer_tokens
