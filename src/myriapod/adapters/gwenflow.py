"""Gwenflow adapter: run myriapod with gwenflow Agents as planner/workers.

Requires the optional extra: ``uv add myriapod[gwenflow]`` (or having
gwenflow installed). Everything is soft-imported so the core never depends
on it.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

try:
    from gwenflow.agents import Agent as _GAgent
    from gwenflow.tools import Tool as _GTool
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The Gwenflow adapter requires the 'gwenflow' package: "
        "install with `uv add 'myriapod[gwenflow]'`."
    ) from e

from myriapod.core.planner import PLANNER_SYSTEM_PROMPT
from myriapod.core.protocol import RunOutcome
from myriapod.core.reviewer import REVIEWER_INSTRUCTIONS
from myriapod.core.scheduler import Swarm
from myriapod.core.task_tree import EFFORTS, TaskNode
from myriapod.core.worker import WORKER_INSTRUCTIONS

logger = logging.getLogger("myriapod")


#: Anthropic's SDK refuses a *non-streaming* request whose predicted runtime
#: passes ten minutes — ``3600 * max_tokens / 128000 > 600``, i.e. any
#: ``max_tokens`` above this. gwenflow calls the API non-streaming, so a
#: generous ceiling meant to stop a plan being truncated instead makes every
#: planner turn fail before it starts. Warned about at build time: the run
#: that discovered this died on three consecutive turns having planned
#: nothing, and the error names streaming rather than max_tokens.
#:
#: It is a *client-side* guard against an idle connection being dropped, not
#: an API limit, and it is suppressed by setting an explicit ``timeout`` on
#: the LLM — which gwenflow passes straight to the Anthropic client. So the
#: ceiling binds only when no timeout is set, and a tier that genuinely needs
#: more room (a thinking model that must reason *and* answer within one
#: budget) buys it with a timeout rather than being stuck here.
ANTHROPIC_NONSTREAMING_MAX_TOKENS = 21_333


def _warn_if_unreachable(llm: Any, role: str) -> None:
    max_tokens = getattr(llm, "max_tokens", None) or 0
    if getattr(llm, "timeout", None):
        return  # An explicit timeout suppresses the SDK's long-request guard.
    if max_tokens > ANTHROPIC_NONSTREAMING_MAX_TOKENS and "anthropic" in type(
        llm
    ).__name__.lower():
        logger.warning(
            "%s max_tokens=%d exceeds the %d ceiling for non-streaming "
            "Anthropic calls; every %s call will fail with 'Streaming is "
            "required'. Lower it, or set an explicit timeout on the LLM to "
            "suppress the SDK guard.",
            role, max_tokens, ANTHROPIC_NONSTREAMING_MAX_TOKENS, role,
        )


class GwenflowAgent:
    """Wrap a ``gwenflow.agents.Agent`` into myriapod's :class:`AgentLike`."""

    def __init__(self, agent: _GAgent, name: str = "gwenflow"):
        self._agent = agent
        self.name = name

    async def run(self, message: str, context: str | None = None) -> RunOutcome:
        response = await self._agent.arun(message, context=context)
        usage = getattr(response, "usage", None)
        content = getattr(response, "content", None) or ""
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
        return RunOutcome(
            content=content,
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=output_tokens,
            empty_reason=(
                None if content else self._empty_reason(output_tokens)
            ),
        )

    def _empty_reason(self, output_tokens: int) -> str | None:
        """Name the cause when the agent returned no text at all.

        One cause dominates on a thinking model and is invisible from the
        core: the response was cut off *inside the thinking block*, before a
        word of text or a single tool call. Anthropic then returns a lone
        thinking block with ``stop_reason: "max_tokens"``, gwenflow drops the
        stop reason on the floor (``_parse_response`` only distinguishes
        ``tool_use``), and its agent loop reads "no tool calls" as "the agent
        is done" — so it breaks after exactly one call with ``content=None``.
        The tell is the token count landing exactly on ``max_tokens``.

        Left undiagnosed this reads as "the task is too big for one answer",
        and the fix that follows from it — split the brief — is wrong: the
        brief was never read out loud. The fix is to give the model room to
        think, which on this path means a larger ``max_tokens`` (and the
        explicit ``timeout`` that lets it exceed
        :data:`ANTHROPIC_NONSTREAMING_MAX_TOKENS`).
        """
        max_tokens = getattr(getattr(self._agent, "llm", None), "max_tokens", None)
        if max_tokens and output_tokens >= max_tokens:
            return (
                f"Worker hit its output ceiling ({output_tokens} of "
                f"max_tokens={max_tokens}) before producing any text or tool "
                "call — the whole budget went to reasoning. Raise max_tokens "
                "for this tier rather than splitting the task; the brief was "
                "never answered, not answered badly."
            )
        return None


def build_swarm(
    planner_llm: Any,
    worker_llm: Any | dict[str, Any],
    *,
    reviewer_llm: Any | None = None,
    worker_tools: Sequence[Any] | None = None,
    planner_system_prompt: str = PLANNER_SYSTEM_PROMPT,
    worker_system_prompt: str = WORKER_INSTRUCTIONS,
    reviewer_system_prompt: str = REVIEWER_INSTRUCTIONS,
    cache_planner_prompt: bool = True,
    **swarm_kwargs: Any,
) -> Swarm:
    """A ready-to-run :class:`Swarm` on gwenflow LLM instances.

    A fresh worker Agent is built per task (gwenflow agents keep history),
    which also enables per-task specialization.

    ``worker_llm`` may be a **fleet**: ``{"low": llm, "standard": llm,
    "high": llm}`` routes each task to the tier its brief declared (see
    ``TaskNode.effort``). Missing tiers fall back to "standard".

    ``reviewer_llm`` turns on the quality gate — that model reads every
    finished answer against its brief and can send it back for a rewrite.
    Point it at a cheap model: it roughly doubles the calls per task.

    ``cache_planner_prompt`` marks the planner's system prompt as cacheable.
    The planner is re-sent its whole prompt (and a growing tree) at every
    turn, which is the one repeated payload in a run; on Anthropic a cache
    read is a tenth of the price of the same tokens read fresh.
    """
    fleet = _fleet(worker_llm)
    _warn_if_unreachable(planner_llm, "planner")
    for effort, llm in fleet.items():
        _warn_if_unreachable(llm, f"worker[{effort}]")

    def planner_factory(tools: list[Callable]) -> GwenflowAgent:
        llm = planner_llm
        if cache_planner_prompt and hasattr(llm, "cache_system_prompt"):
            # Copied rather than mutated: the caller's object may well be
            # reused for another swarm, and silently flipping a flag on it
            # is the kind of action at a distance that is impossible to
            # debug from the outside.
            llm = _copy(llm)
            llm.cache_system_prompt = True
        agent = _GAgent(
            name="myriapod-planner",
            system_prompt=planner_system_prompt,
            llm=llm,
            tools=[_GTool(fn) for fn in tools],
        )
        return GwenflowAgent(agent, name="planner[gwenflow]")

    def worker_factory(task: TaskNode) -> GwenflowAgent:
        effort = getattr(task, "effort", "standard")
        llm = fleet.get(effort) or fleet["standard"]
        agent = _GAgent(
            name=f"myriapod-worker-{task.id}",
            instructions=worker_system_prompt,
            llm=_copy(llm),
            tools=list(worker_tools or []),
        )
        return GwenflowAgent(agent, name=f"worker-{task.id}[{effort}]")

    if reviewer_llm is not None:
        def reviewer_factory(task: TaskNode) -> GwenflowAgent:
            agent = _GAgent(
                name=f"myriapod-reviewer-{task.id}",
                instructions=reviewer_system_prompt,
                llm=_copy(reviewer_llm),
            )
            return GwenflowAgent(agent, name=f"reviewer-{task.id}[gwenflow]")

        swarm_kwargs.setdefault("reviewer_factory", reviewer_factory)

    return Swarm(
        planner_factory=planner_factory,
        worker_factory=worker_factory,
        **swarm_kwargs,
    )


def _copy(llm: Any) -> Any:
    """A private copy of an LLM config, since gwenflow agents mutate theirs."""
    return llm.model_copy(deep=True) if hasattr(llm, "model_copy") else llm


def _fleet(worker_llm: Any | dict[str, Any]) -> dict[str, Any]:
    """Normalize one LLM or a per-effort mapping into a full fleet."""
    if not isinstance(worker_llm, dict):
        return {effort: worker_llm for effort in EFFORTS}
    if not worker_llm:
        raise ValueError("worker_llm mapping is empty.")
    unknown = set(worker_llm) - set(EFFORTS)
    if unknown:
        raise ValueError(
            f"Unknown effort tier(s) {sorted(unknown)}; expected {list(EFFORTS)}."
        )
    standard = worker_llm.get("standard") or next(iter(worker_llm.values()))
    return {effort: worker_llm.get(effort, standard) for effort in EFFORTS}
