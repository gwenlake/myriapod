"""Optional quality gate between a worker's answer and the tree.

The planner reads summaries, not answers — by design. That is what keeps a
thousand-task run inside one context window, and it is also the blind spot:
a worker that answers a five-sentence brief with three vague paragraphs
writes a perfectly confident summary about it, and nothing downstream ever
notices. The gap is only visible to a reader holding *both* the brief and
the answer, which is nobody in the base loop.

A reviewer is that reader. It is a cheap model given the brief and the
answer, asked one question — does this deliver what was asked? — and
allowed to answer only PASS or REVISE with reasons. On REVISE the same task
runs again with the reasons attached, which is far cheaper than a planner
turn spent discovering the same thing two waves later.

Fail-open on purpose: an unparseable verdict counts as PASS. A judge is an
optimisation, and an optimisation that can wedge a run is a liability.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

from myriapod.core.protocol import AgentLike

logger = logging.getLogger("myriapod")

REVIEWER_INSTRUCTIONS = """\
You are a REVIEWER in an agent swarm. You are given the brief that was sent
to a worker and the answer it produced. You judge whether the answer
delivers the brief. You never rewrite it and you never do the task yourself.

Judge only against the brief, on five points:

1. Coverage — is every element the brief asked for actually present? A brief
   asking for 12 ideas and answered with 6 fails, however good the 6 are.
2. Specificity — named actors, dates, quantities with units, prices,
   concrete examples. A sentence that would be true of any company in the
   sector is filler, not content.
3. Substance — is the reasoning behind non-obvious claims given, or only the
   claims? Are stated uncertainties honest rather than hedging everywhere?
4. Form — the requested format, structure and rough length. Markdown, the
   sections asked for, the table if one was asked for.
5. Integrity — is the answer complete, or does it stop mid-sentence, leave a
   TBD, or promise a section it never writes?

Be demanding but proportionate: ask for a revision when a worker could
plausibly do materially better on a re-run, not because a perfect answer
would have been longer. A revision costs a full re-run, so REVISE must
point at something specific and fixable.

Answer in exactly this shape, and nothing else:

VERDICT: PASS
or
VERDICT: REVISE
THEN, only when revising, 2 to 5 numbered lines, each naming one concrete
gap and what the worker must add or change. Address the worker directly and
quote the part of the brief it missed. No praise, no summary of the answer,
no rewriting."""

_VERDICT_RE = re.compile(r"VERDICT:\s*(PASS|REVISE)", re.IGNORECASE)


@dataclass(slots=True)
class Verdict:
    """One review: whether to accept the answer, and why not."""

    passed: bool
    feedback: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


def build_review_input(brief: str, answer: str) -> tuple[str, str]:
    """(message, context) sent to the reviewer for one answer."""
    message = (
        "Judge the answer below against its brief. Reply with VERDICT: PASS "
        "or VERDICT: REVISE followed by the numbered gaps."
    )
    context = (
        f"### The brief the worker was given\n{brief}\n\n"
        f"### The answer it produced\n{answer}"
    )
    return message, context


def parse_verdict(content: str) -> tuple[bool, str]:
    """(passed, feedback) from a reviewer answer; unreadable means passed."""
    match = _VERDICT_RE.search(content or "")
    if match is None:
        return True, ""
    if match.group(1).upper() == "PASS":
        return True, ""
    feedback = content[match.end():].strip()
    # A REVISE with no reason is not actionable — re-running the worker with
    # "do better" attached buys nothing but a second bill.
    return (False, feedback) if feedback else (True, "")


async def review_answer(
    reviewer: AgentLike, brief: str, answer: str, timeout: float | None = None
) -> Verdict:
    """Run one review. Any failure of the reviewer is a PASS."""
    message, context = build_review_input(brief, answer)
    try:
        coro = reviewer.run(message, context=context)
        outcome = await (asyncio.wait_for(coro, timeout) if timeout else coro)
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        logger.warning("reviewer failed, accepting the answer: %s: %s", type(e).__name__, e)
        return Verdict(passed=True)

    passed, feedback = parse_verdict(getattr(outcome, "content", "") or "")
    return Verdict(
        passed=passed,
        feedback=feedback,
        input_tokens=getattr(outcome, "input_tokens", 0),
        output_tokens=getattr(outcome, "output_tokens", 0),
    )


def revision_note(feedback: str, previous: str) -> str:
    """What gets appended to a worker's context when it must try again."""
    return (
        "### Your previous answer to this task\n"
        f"{previous}\n\n"
        "### Review of that answer — it was rejected\n"
        f"{feedback}\n\n"
        "Write the deliverable again, in full, fixing every point above. "
        "This is a rewrite, not a patch: produce the complete answer, keep "
        "what was good, and do not mention the review or the previous "
        "attempt. End with the <summary> block as usual."
    )
