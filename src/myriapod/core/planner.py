"""Planner side of the swarm: system prompt and tree-piloting tools.

The planner is a regular gwenflow ``Agent`` on a frontier model, given four
tools that mutate the shared :class:`TaskTree`. It never executes work and
never sees raw worker output — only ``tree.render()``. Tool validation
errors are returned as text so the model can self-correct within its turn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from myriapod.core.task_tree import TaskTree, TaskTreeError
from myriapod.core.worker import TRUNCATED_MARKER

PLANNER_SYSTEM_PROMPT = """\
You are the PLANNER of a swarm of AI agents. You decompose a goal into a
tree of tasks; cheap, fast WORKER agents execute the leaves in parallel.
You never do the work yourself and you never write deliverable content:
you plan, monitor, and decide.

A worker sees ONLY the task description you write plus the full outputs of
the dependencies you declare. It cannot see the goal, the other tasks, your
reasoning, or ask you anything. A vague description is the single biggest
cause of shallow output, so writing good ones is the highest-value thing
you do with a turn.

## Writing a task description

Each entry you pass to plan_tasks is a standalone brief. Cover, in prose:

- Objective — the one question this task answers, or the artefact it
  produces.
- Substance — what specifically to cover: the mechanisms, actors, periods,
  figures or trade-offs that must appear. Name them; do not write "the
  relevant details". Spell out acronyms, since the worker has no context to
  resolve them from.
- Boundaries — what to leave out and who owns it: "Do not cover the
  manufacturing process, task 4 owns it." This is what stops several
  workers writing the same paragraph.
- Deliverable — format and rough size: "~600 words of Markdown, one ##
  section per sub-question", "a table with one row per model". Workers
  write Markdown, and the final answer may be exported to a document or a
  slide deck.
- Quality bar — the precision expected: dates, quantities with units, named
  sources, the reasoning behind a claim rather than the claim alone. Say
  where a statement must be attributed and where uncertainty must be
  stated as such.

A one-line description produces a one-paragraph answer. A brief that
actually gets depth is usually three to six sentences.

One task is one response from one worker, so it has a ceiling: ask for more
than roughly 1200 words in a single task and the answer comes back cut off
mid-sentence. Split anything larger across tasks. If a summary ever comes
back marked as having no summary block, that task was truncated — re-plan it
as smaller pieces rather than shipping it.

## How much to decompose

Match the fan-out to the goal, and prefer wide and flat over deep:

- a single fact or a short answer — 1 task;
- a comparison or a two-part question — 2 to 4 tasks, one per element;
- a report, a study, a survey — 6 to 15 tasks, one per section or angle,
  plus a synthesis task;
- a large enumeration (per country, per file, per product) — one task per
  item, across as many plan_tasks calls as it takes.

Independent tasks run concurrently, so breadth costs almost nothing in wall
clock. Depth does, because every dependency serialises a chain: declare one
only when a task genuinely needs another's output.

**Send at most 8 subtasks per plan_tasks call.** The briefs you write are
output tokens, and a tool call has a hard output ceiling: past a handful of
detailed briefs the call is cut off mid-argument and the whole batch is
lost — you get nothing back and have to write it all again. To plan twenty
tasks, call plan_tasks three times with the same parent_id: each call
appends new siblings, and depends_on in a later call can reference the ids
the earlier calls returned. Several small calls in one turn are strictly
better than one big one.

## Choosing the effort

Each subtask may declare an effort — "low", "standard" (the default) or
"high" — and the fleet routes on it: low goes to the cheapest model
available, high to a stronger one. Spend high where the deliverable's
quality actually rests on judgement across many inputs: synthesis,
prioritisation, arbitration between contradictory findings, anything
adversarial. Spend low on mechanical work: extraction, listing, formatting,
reformulation. Everything else is standard.

Marking everything high does not make the answer better, it makes the run
slow and expensive; marking everything low makes a cheap model arbitrate
your hardest question. Both are worse than choosing.

## Assembling the answer

For anything longer than a couple of paragraphs, create a final synthesis
task that depends on the parts. Give it the outline you want and tell it to
reconcile contradictions rather than concatenate. Then use
finish(from_task_id=...) so its full output becomes the answer. Write
final_answer yourself only for trivial glue.

## Your turns

You are called when the frontier is exhausted: at the start, when the swarm
is blocked, and when everything is settled. Read the tree, then act.

- retry_task for a transient failure; skip_task for a dead end, with the
  reason.
- plan_tasks to extend or repair the plan — including follow-up tasks when
  a summary reveals a gap, a contradiction between two workers, or an
  answer thinner than the brief asked for.
- finish when the goal is genuinely satisfied. Before you do, check the
  summaries against the goal and cover what is missing: a thin answer
  delivered early is worse than one more round of tasks.

Task summaries are digests — trust them for status and coverage, and rely
on dependencies to move full content between workers. You interact ONLY
through your tools."""


#: Returned when `subtasks` arrives missing or unterminated. Written for the
#: planner: name the cause, then give the one call that gets out of it.
_TRUNCATED_ARGS_HINT = (
    "No usable subtasks in that call: its arguments were cut off by the "
    "model output limit — you wrote more briefs than one tool call can "
    "carry, so nothing was created. Call plan_tasks again with AT MOST 5 "
    "subtasks, then call it again for the next batch (same parent_id: the "
    "new tasks are appended as siblings, and depends_on may reference the "
    "ids returned by an earlier batch). Do not resend the whole plan in one "
    "call — it will be cut off again."
)


def _looks_truncated(raw: Any) -> bool:
    """Is this JSON string an unterminated fragment rather than a typo?"""
    if not isinstance(raw, str):
        return False
    text = raw.strip()
    return bool(text) and not text.endswith(("]", "}"))


@dataclass
class SwarmState:
    """Mutable flags shared between planner tools and the orchestrator."""

    finished: bool = False
    final_answer: str | None = None
    final_task_id: str | None = None
    planned_something: bool = False
    log: list[dict[str, Any]] = field(default_factory=list)
    #: Task ids whose truncated output ``finish`` has already pushed back on
    #: once. Asking twice is taken as a deliberate choice, not an oversight.
    truncation_warned: set[str] = field(default_factory=set)


def make_planner_tools(
    tree: TaskTree, state: SwarmState
) -> list[Callable[..., str]]:
    """Build the planner tool functions (closures over the shared tree).

    Returned as plain callables; the swarm wraps them in ``gwenflow.tools.Tool``
    so this module stays import-light and easy to test.
    """

    def plan_tasks(
        parent_id: str, subtasks: list[dict[str, Any]] | str | None = None
    ) -> str:
        """Decompose a task into subtasks executed by workers.

        Args:
            parent_id: Id of the task to decompose. Use "root" for the goal.
            subtasks: Required. Array of at most 8 objects, each with
                "description" (string, complete and self-contained), optional
                "depends_on" (array of sibling task ids, or 1-based indices
                within this array) and optional "effort" ("low", "standard"
                or "high" — which model tier executes it). A JSON string
                encoding that array is accepted too. Call plan_tasks again
                for the next batch.
        """
        # A missing or half-written `subtasks` is almost never a modelling
        # mistake: the model wrote so many briefs that the tool call itself
        # hit the output ceiling, and the provider dropped (Anthropic) or
        # mangled (partial JSON) the unfinished argument. Saying so is what
        # turns a dead run into a smaller second call — the raw framework
        # error ("missing 1 required positional argument") tells the planner
        # nothing it can act on, so it retried the same oversized call.
        if subtasks is None or (isinstance(subtasks, str) and not subtasks.strip()):
            return _TRUNCATED_ARGS_HINT
        try:
            parsed = json.loads(subtasks) if isinstance(subtasks, str) else subtasks
            if isinstance(parsed, dict):
                parsed = [parsed]
            if not isinstance(parsed, list):
                raise TaskTreeError("subtasks must be an array of objects.")
            created = tree.decompose(parent_id, parsed)
        except json.JSONDecodeError as e:
            return (
                f"Invalid JSON for subtasks: {e}. {_TRUNCATED_ARGS_HINT}"
                if _looks_truncated(subtasks)
                else f"Invalid JSON for subtasks: {e}. Fix the JSON and call "
                     "plan_tasks again."
            )
        except TaskTreeError as e:
            return f"Planning error: {e}"
        state.planned_something = True
        state.log.append({"action": "plan", "parent": parent_id, "count": len(created)})
        ids = ", ".join(n.id for n in created)
        return f"Created {len(created)} task(s): {ids}. Workers will start on the ready ones."

    def retry_task(task_id: str) -> str:
        """Reset a failed task so a fresh worker retries it.

        Args:
            task_id: Id of the failed leaf task to retry.
        """
        try:
            tree.reset_task(task_id)
        except TaskTreeError as e:
            return f"Planning error: {e}"
        state.planned_something = True
        state.log.append({"action": "retry", "task": task_id})
        return f"Task {task_id} reset; a worker will retry it."

    def skip_task(task_id: str, reason: str) -> str:
        """Abandon a task that is impossible or no longer useful.

        Args:
            task_id: Id of the leaf task to skip.
            reason: Short explanation, recorded in the tree.
        """
        try:
            tree.skip_task(task_id, reason)
        except TaskTreeError as e:
            return f"Planning error: {e}"
        state.planned_something = True
        state.log.append({"action": "skip", "task": task_id, "reason": reason})
        return f"Task {task_id} skipped."

    def finish(final_answer: str = "", from_task_id: str = "") -> str:
        """End the swarm run with the final answer to the goal.

        Args:
            final_answer: The final answer text, if you write it yourself.
            from_task_id: Id of a completed task whose full output should be
                promoted as the final answer (preferred for long deliverables).
        """
        # A planner that calls plan_tasks and finish in the same turn ends the
        # run before a single worker has been dispatched: the scheduler stops
        # as soon as `finished` is set, so the answer would be the planner's
        # own guess and the plan would never execute. Refuse and let the
        # frontier drain; the planner is called again once it is empty.
        by_status = tree.stats()["by_status"]
        outstanding = by_status.get("pending", 0) + by_status.get("in_progress", 0)
        if outstanding:
            return (
                f"Not yet: {outstanding} task(s) are still pending or running, "
                "and their output does not exist yet. Do not finish now — end "
                "this turn instead. You will be called again with the results "
                "once the frontier is empty, and can finish then."
            )
        if from_task_id:
            try:
                node = tree.get(from_task_id)
            except TaskTreeError as e:
                return f"Planning error: {e}"
            if not node.result_full and not node.result_summary:
                return (
                    f"Task {from_task_id} has no output yet "
                    f"(status: {node.status.value}). Wait for it or answer directly."
                )
            # Promoting a truncated task makes the deliverable stop
            # mid-sentence — the precise outcome the truncation marker exists
            # to prevent, arriving through the one door that never checked it.
            # Pushed back once, not forbidden: after a re-plan the planner may
            # legitimately decide a long cut-off answer beats the alternative,
            # and a hard refusal would leave a turn-starved run with nothing
            # but summaries.
            if (node.result_summary or "").startswith(TRUNCATED_MARKER):
                if from_task_id not in state.truncation_warned:
                    state.truncation_warned.add(from_task_id)
                    return (
                        f"Task {from_task_id} was cut off at the model's "
                        "output limit: its text stops mid-sentence, and "
                        "shipping it as the final answer ships that. Re-plan "
                        "it as two or three smaller tasks and assemble them, "
                        "or retry_task it with a narrower brief. If you have "
                        "considered it and still want this text as-is, call "
                        "finish with the same from_task_id again."
                    )
                state.log.append({"action": "finish_truncated", "task": from_task_id})
            state.final_answer = node.result_full or node.result_summary
            state.final_task_id = from_task_id
        elif final_answer.strip():
            state.final_answer = final_answer.strip()
        else:
            return "Provide final_answer or from_task_id."
        state.finished = True
        state.log.append({"action": "finish", "from_task": from_task_id or None})
        return "Swarm run completed."

    return [plan_tasks, retry_task, skip_task, finish]


def planner_turn_message(tree: TaskTree, note: str | None = None) -> str:
    """The message sent to the planner at each turn: the rendered tree."""
    parts = ["Current task tree:", "", tree.render(), ""]
    if note:
        parts += [note, ""]
    parts.append(
        "Decide the next move: plan_tasks / retry_task / skip_task, "
        "or finish if the goal is satisfied."
    )
    return "\n".join(parts)
