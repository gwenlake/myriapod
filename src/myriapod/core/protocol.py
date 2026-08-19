"""The minimal contract between myriapod and any agent framework.

myriapod's core never imports an agent framework. It drives anything that
satisfies :class:`AgentLike` and returns a :class:`RunOutcome`. Adapters
(``myriapod.adapters.*``) wrap Pydantic AI, Gwenflow, or your own runtime
into this contract — usually in a couple dozen lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(slots=True)
class RunOutcome:
    """Normalized result of one agent run."""

    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    #: Why the answer came back empty, when the adapter can tell. An empty
    #: answer has several causes that look identical from the core — the call
    #: never ran, the agent burned its budget on tool calls, or the model hit
    #: its output ceiling before writing a word — and only the adapter holds
    #: the model settings needed to tell them apart. Left ``None``, the core
    #: reports the token count and lets the reader infer; set, it is quoted
    #: verbatim, because a wrong guess here sends the planner off rewriting a
    #: brief that was never the problem.
    empty_reason: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@runtime_checkable
class AgentLike(Protocol):
    """Anything that can take a prompt and return a :class:`RunOutcome`.

    ``name`` is optional (used in logs and the task tree); implement it as
    an attribute or property when you can.
    """

    async def run(self, message: str, context: str | None = None) -> RunOutcome: ...


class PlannerFactory(Protocol):
    """Builds the planner agent for one run.

    Receives the tree-piloting tool *functions* (plain callables with
    google-style docstrings) and must return an agent that exposes them to
    the model. Called once per :meth:`Swarm.arun`, because the tools close
    over that run's task tree.
    """

    def __call__(self, tools: list) -> AgentLike: ...


class WorkerFactory(Protocol):
    """Returns the agent to execute one leaf task.

    Called for every task, so it may return a shared, reusable agent (cheap
    and fine when runs are stateless, as in Pydantic AI) or build a
    specialized one per task (heterogeneous fleets).
    """

    def __call__(self, task) -> AgentLike: ...
