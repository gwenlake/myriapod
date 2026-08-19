"""Pydantic AI adapter: run myriapod on any model Pydantic AI supports.

Model strings use Pydantic AI's ``provider:model`` form, e.g.
``"openai:gpt-5"``, ``"anthropic:claude-haiku-4-5"``, ``"google-gla:..."``,
``"ollama:..."``. API keys come from the usual environment variables.

The planner's tree-piloting tools are plain Python callables with
google-style docstrings — exactly what Pydantic AI consumes natively, so
they are passed straight through to ``Agent(tools=...)``.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

try:
    from pydantic_ai import Agent as _PAgent
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The Pydantic AI adapter requires the 'pydantic-ai' package. "
        "Install myriapod with its default dependencies (uv sync) or "
        "`uv add pydantic-ai`."
    ) from e

from myriapod.core.planner import PLANNER_SYSTEM_PROMPT
from myriapod.core.protocol import RunOutcome
from myriapod.core.reviewer import REVIEWER_INSTRUCTIONS
from myriapod.core.scheduler import Swarm
from myriapod.core.task_tree import EFFORTS, TaskNode
from myriapod.core.worker import WORKER_INSTRUCTIONS


def _extract_outcome(result: Any) -> RunOutcome:
    """Normalize a Pydantic AI run result across versions."""
    content = getattr(result, "output", None)
    if content is None:
        content = getattr(result, "data", None)  # pre-1.0 spelling
    usage = getattr(result, "usage", None)
    if callable(usage):
        usage = usage()
    input_tokens = (
        getattr(usage, "input_tokens", None)
        or getattr(usage, "request_tokens", None)
        or 0
    )
    output_tokens = (
        getattr(usage, "output_tokens", None)
        or getattr(usage, "response_tokens", None)
        or 0
    )
    return RunOutcome(
        content=str(content) if content is not None else "",
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
    )


class PydanticAIAgent:
    """Wrap a ``pydantic_ai.Agent`` into myriapod's :class:`AgentLike`."""

    def __init__(self, agent: _PAgent, name: str = "pydantic-ai"):
        self._agent = agent
        self.name = name

    async def run(self, message: str, context: str | None = None) -> RunOutcome:
        prompt = message if not context else f"{message}\n\n{context}"
        result = await self._agent.run(prompt)
        return _extract_outcome(result)


#: Output-token ceiling given to both roles unless the caller overrides it.
#: Pydantic AI's own default is low enough (4096 on Anthropic) that a worker
#: writing a report section, or a planner writing a fifteen-task plan, gets
#: cut off mid-answer — and a truncated answer looks exactly like a finished
#: one to everything downstream.
DEFAULT_MAX_TOKENS = 8192


#: A review is a verdict and a handful of numbered gaps, never prose.
DEFAULT_REVIEW_MAX_TOKENS = 1024


def build_swarm(
    planner_model: str,
    worker_model: str | dict[str, str],
    *,
    reviewer_model: str | None = None,
    worker_tools: Sequence[Callable] | None = None,
    planner_system_prompt: str = PLANNER_SYSTEM_PROMPT,
    worker_system_prompt: str = WORKER_INSTRUCTIONS,
    reviewer_system_prompt: str = REVIEWER_INSTRUCTIONS,
    planner_max_tokens: int | None = DEFAULT_MAX_TOKENS,
    worker_max_tokens: int | None = DEFAULT_MAX_TOKENS,
    reviewer_max_tokens: int | None = DEFAULT_REVIEW_MAX_TOKENS,
    planner_agent_kwargs: dict[str, Any] | None = None,
    worker_agent_kwargs: dict[str, Any] | None = None,
    **swarm_kwargs: Any,
) -> Swarm:
    """A ready-to-run :class:`Swarm` on Pydantic AI model strings.

    The worker agent is built once per model and shared by the whole fleet —
    Pydantic AI agents are stateless across ``run()`` calls (FastAPI-app
    style), so reuse is safe and keeps 1000-task runs cheap to schedule.
    ``**swarm_kwargs`` are forwarded to :class:`Swarm` (max_concurrency,
    max_cost, worker_timeout, pricing tables...).

    ``worker_model`` may be a **fleet**: pass ``{"low": ..., "standard":
    ..., "high": ...}`` and each task goes to the tier its brief declared
    (see ``TaskNode.effort``). Missing tiers fall back to "standard", so
    ``{"standard": cheap, "high": strong}`` is a complete fleet. One string
    means one model for everything, as before.

    ``reviewer_model`` turns on the quality gate: that model reads each
    finished answer against its brief and can send it back for one rewrite
    (``max_revisions`` in ``swarm_kwargs``). It roughly doubles the calls per
    task, so point it at a cheap model.

    ``planner_max_tokens`` / ``worker_max_tokens`` set the output ceiling
    through ``model_settings``; pass ``None`` to leave the provider default
    alone. An explicit ``model_settings`` in the ``*_agent_kwargs`` wins.
    """

    def with_max_tokens(
        kwargs: dict[str, Any] | None, max_tokens: int | None
    ) -> dict[str, Any]:
        kwargs = dict(kwargs or {})
        if max_tokens is not None and "model_settings" not in kwargs:
            kwargs["model_settings"] = {"max_tokens": max_tokens}
        return kwargs

    def planner_factory(tools: list) -> PydanticAIAgent:
        # Pydantic AI defaults to a single retry on a tool-argument validation
        # failure, then raises and takes the whole run down with it. A planner
        # writing a 15-task plan in one call gets one malformed argument often
        # enough that one retry is not enough.
        kwargs = {
            "retries": 3,
            **with_max_tokens(planner_agent_kwargs, planner_max_tokens),
        }
        agent = _PAgent(
            planner_model,
            system_prompt=planner_system_prompt,
            tools=list(tools),
            **kwargs,
        )
        return PydanticAIAgent(agent, name=f"planner[{planner_model}]")

    fleet = {
        effort: PydanticAIAgent(
            _PAgent(
                model,
                system_prompt=worker_system_prompt,
                tools=list(worker_tools or []),
                **with_max_tokens(worker_agent_kwargs, worker_max_tokens),
            ),
            name=f"worker[{model}]",
        )
        for effort, model in _fleet_models(worker_model).items()
    }

    def worker_factory(task: TaskNode) -> PydanticAIAgent:
        # An unknown or absent tier is not an error: a planner may declare
        # "high" on a run configured with one model, and that task must still
        # run rather than fail on a KeyError.
        return fleet.get(getattr(task, "effort", "standard")) or fleet["standard"]

    if reviewer_model is not None:
        reviewer = PydanticAIAgent(
            _PAgent(
                reviewer_model,
                system_prompt=reviewer_system_prompt,
                **with_max_tokens(None, reviewer_max_tokens),
            ),
            name=f"reviewer[{reviewer_model}]",
        )
        swarm_kwargs.setdefault("reviewer_factory", lambda task: reviewer)

    return Swarm(
        planner_factory=planner_factory,
        worker_factory=worker_factory,
        **swarm_kwargs,
    )


def _fleet_models(worker_model: str | dict[str, str]) -> dict[str, str]:
    """Normalize a model string or per-effort mapping into a full fleet.

    Always yields a "standard" entry: it is the fallback every other tier
    resolves to, so a partial mapping stays usable.
    """
    if isinstance(worker_model, str):
        return {effort: worker_model for effort in EFFORTS}
    if not worker_model:
        raise ValueError("worker_model mapping is empty.")
    unknown = set(worker_model) - set(EFFORTS)
    if unknown:
        raise ValueError(
            f"Unknown effort tier(s) {sorted(unknown)}; expected {list(EFFORTS)}."
        )
    standard = worker_model.get("standard") or next(iter(worker_model.values()))
    return {effort: worker_model.get(effort, standard) for effort in EFFORTS}
