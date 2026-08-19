# CLAUDE.md

Guidance for Claude Code working in this repo. `README.md` is the user-facing
doc; this file is what is not obvious from reading the code.

## What this is

A planner/worker agent swarm over a dynamic task tree: one frontier model
plans, a fleet of cheap models executes the leaves in parallel. The core is
framework-free; adapters bind it to a runtime (Pydantic AI by default).

## Commands

```bash
uv sync --all-extras                      # install (see the note below)
uv run python -m pytest                   # 109 tests, all offline, no API keys
uv run myriapod bench -n 1000 -c 500      # scheduler benchmark, offline, free
uv run myriapod bench -n 40 --waves 3 --groups 4 --planner-delay 2 --viz
uv run myriapod ask "..."                 # real run (Opus 5 + Haiku 4.5, $1.5 cap)
```

Two things that look like bugs and are not:

- **`uv sync --all-extras`, not a bare `uv sync`.** A bare sync (or a bare
  `uv run`) can prune the optional extras, which surfaces later as
  `No module named 'fpdf'` in the report tests — a failure that names a
  dependency rather than the sync that dropped it.
- **`uv run python -m pytest`, not `uv run pytest`.** The dev group installs
  the pytest *module* but no console script in this environment, so the bare
  form dies on `Failed to spawn: pytest`.

Prefer `bench` over `ask` when validating scheduler or viz changes: it drives
the whole machine with simulated agents, so it costs nothing and is
deterministic enough to compare runs.

## Architecture invariants

**Context isolation is the point of the project.** Breaking it breaks the
scaling claim, so treat these as load-bearing:

- The planner only ever sees `tree.render()` — statuses and short summaries.
  It must never receive `result_full`.
- A worker sees its task description plus the outputs of its **direct
  declared dependencies**, and nothing else. Full content moves only along
  dependency edges.
- Those dependency outputs are **fitted to a character budget**
  (`worker.build_worker_input`, `DEFAULT_CONTEXT_CHARS`), not concatenated:
  a synthesis task declaring twenty dependencies used to receive twenty
  whole answers, so the context grew with the fan-out — the exact thing the
  isolation claim says it does not. Allocation is fair share with spillover,
  and a dependency granted less than `MIN_EXCERPT_CHARS` is passed as its
  digest instead: 1kB of a report is its introduction, and a worker reading
  forty introductions is worse informed than one reading forty digests while
  believing itself better informed. Demoting one dependency frees budget and
  may lift another over the floor, which is why `_allocate` loops.
- The viz wire carries truncated descriptions and summaries only
  (`viz/__init__.py::_node_payload`). `result_full` must not leave the
  process.

**Scheduler** (`core/scheduler.py`) — the planner is consulted only when the
frontier is exhausted (start, blocked, done); worker completions refill the
pool via `asyncio.wait(FIRST_COMPLETED)`. Transient worker failures are
retried with backoff *inside* the worker coroutine, so only persistent
failures reach a planner turn. Don't add per-batch synchronisation.

A planner turn runs **concurrently with dispatch**: it is an `asyncio.Task`,
and its `plan_tasks` calls mutate the tree as it makes them, so workers start
on the first briefs while it is still writing the rest. Awaiting the turn
instead left a fifty-task fan-out idle for the ~9 minutes the planner took.
Nothing wakes the loop when a tool call lands (they can run in a worker
thread), so while a turn is in flight the wait carries a
`PLANNER_POLL_SECONDS` timeout and an empty `done` set just means "re-check
the frontier". Keep that timeout if you touch the loop — without it the
concurrency silently disappears, which is exactly the bug
`test_workers_start_while_the_planner_turn_is_still_running` pins down.

**Iterative planning depends on reopening a settled container.** Planning
into a node whose children have all settled is a *second wave* — the planner
read the summaries and adds follow-ups — so `decompose` reopens it and walks
the ancestors' counters back. Only a settled *leaf* is refused: it owns an
output, and turning it into a container would orphan that. This was broken
until `bench --waves` exposed it: the run stalled after the first wave
because the note telling the planner to "plan follow-up tasks" described
something the tree rejected.

**Dispatch order is the critical path, not the id order.** `ready_tasks()`
sorts by the longest chain of tasks waiting downstream of each leaf
(`_chain_lengths_unlocked`, Kahn + one reverse fold, O(N+E)), ties broken on
id. Two consequences to keep: a leaf inherits the queue waiting on its
*container* (nothing depending on "section 3" starts until all of its leaves
are done, so `_priority` walks the ancestors), and a tree with no dependency
at all short-circuits to id order — the flat thousand-task fan-out must not
pay for an ordering that cannot change anything.

**A run can be resumed.** `Swarm.arun(goal, resume=tree_or_dict)` rebuilds
the tree, keeps every settled task's output, and sends the leaves that were
IN_PROGRESS back to PENDING (`reopen_running`) — nothing would ever settle
them otherwise and `is_complete()` would never be true. `to_dict`/`from_dict`
carry `turn` and `planner_turns`: without them a resumed run died on
`mark_planner_thinking` at its first turn.

That is worth nothing unless the caller *persists* the tree, and the tree is
the only place a run's full outputs exist — the viz wire carries summaries
by design, so a process that exits without writing it has destroyed them.
`examples/04_report_and_resume.py` writes it in a `finally` and reads it back
at startup — copy that shape. The run that motivated this stopped on
`max_cost` with 27 tasks done and no way to reach them.

**Failed attempts are billed too** (`mark_failed` takes token counts). A
worker that burns its whole output budget and returns no final text —
common on a wide fan-in — used to cost real money invisibly, so `stats()`
and therefore `max_cost` under-counted a run that was failing expensively.
The error text carries the token count, because that number is the
diagnosis: near zero means the call never ran, thousands means the task is
too big for one answer and must be split rather than retried.

**TaskTree** (`core/task_tree.py`) — completion is tracked with incremental
per-parent counters (`_settled_children`), not by walking the tree. Any new
mutation must keep them consistent through `_on_settled` / `_on_unsettled`,
or `is_complete()` will lie and the run will hang or end early. The tree uses
a `threading.Lock` because agent frameworks may run tool functions in worker
threads while the scheduler reads from the event loop.

**Worker protocol** (`core/worker.py`) — a worker ends its answer with
`<summary>...</summary>`. The summary is what the planner reads; everything
above it is what dependent workers receive. A missing closing tag almost
always means the answer hit the model's output ceiling, so `run_worker`
prefixes the summary with `TRUNCATED_MARKER` and logs a warning — without
that, a report cut off mid-sentence reaches the planner looking finished and
gets shipped. Keep the marker in whatever `tree.render()` shows the planner.

Related: the adapter sets `max_tokens` (`DEFAULT_MAX_TOKENS = 8192`, CLI
`--max-tokens`) because Pydantic AI's default of 4096 truncates both report
sections and large plans.

**`max_tokens` above 21,333 on the gwenflow path needs an explicit
`timeout`.** Anthropic's SDK refuses a *non-streaming* request whose
predicted runtime passes ten minutes (`3600 * max_tokens / 128000 > 600`),
and gwenflow calls the API non-streaming. Raising the planner to 32000 to
stop a plan being truncated therefore made every planner turn fail instantly
with "Streaming is required" — three consecutive turns, nothing planned,
nothing spent, and an error message that names streaming rather than the
setting that caused it. But that guard is *client-side*, protecting against
a dropped idle connection rather than enforcing an API limit, and setting
`timeout=` on the `ChatAnthropic` (gwenflow passes it straight to the
Anthropic client) suppresses it — verified working at `max_tokens=64000`.
Keep that timeout ≥ `worker_timeout`, or the HTTP client cuts the worker
before the scheduler does. `adapters/gwenflow.py` warns at build time
(`ANTHROPIC_NONSTREAMING_MAX_TOKENS`) and stays quiet once a timeout is set.
For the *planner*, the fix for a plan that does not fit is still smaller
`plan_tasks` batches, not a bigger ceiling.

**A thinking model spends `max_tokens` on reasoning before it writes a
word.** Sonnet 5 and Opus 5 reason by default: gwenflow never sends the
`thinking` parameter, and *omitting* it now means "adaptive", not "off" as
it did on the previous generation. `max_tokens` caps reasoning and answer
*together*, so a dense brief can burn the whole budget inside the thinking
block and come back with no text and no tool call. Anthropic reports
`stop_reason: "max_tokens"`; gwenflow's `_parse_response` only ever
distinguishes `tool_use`, so it drops that, and its agent loop reads "no
tool calls" as "the agent is finished" — breaking after exactly one call
with `content=None`. The tell is an output-token count landing *exactly* on
`max_tokens` (a labs run lost two builders at precisely 21,000, twice each).
`GwenflowAgent._empty_reason` names this on `RunOutcome.empty_reason`,
because the core's default guess — "the task is too big for one answer,
split it" — is the opposite of the fix and sends the planner rewriting a
brief the worker never reached. Give the tier room instead
(`BUILDER_MAX_TOKENS` in the labs example).

**The reviewer (`core/reviewer.py`) is opt-in and must fail open.** It is
the only reader in the system holding *both* a brief and the answer to it —
the planner sees summaries by design, which is what makes a twelve-idea
brief answered with six ideas invisible. On REVISE the task is rewritten
once by a *fresh* agent (gwenflow agents keep history; a revision must not
inherit the answer it replaces). Three rules: an unparseable verdict, a
REVISE with no reason, and a failed reviewer all count as PASS; a failed
rewrite keeps the original answer; and the last rewrite is deliberately not
re-reviewed, since with no revision left the verdict could only cost a call.
Every attempt and every review is booked on the task via `_Spend`, at the
rate of whoever ran — a Haiku review billed at the worker's rate is how a
`max_cost` cap starts lying.

**Model routing rides on `TaskNode.effort`** ("low" | "standard" | "high",
declared per subtask by the planner, normalized in `_effort` so an unknown
value is "standard" rather than an error). The tree only records it; a
`worker_factory` turns it into a model. `worker_pricing` therefore accepts
either one table or one per tier — route half a run to a model three times
the price under a single table and the run under-reports itself, which
`resolve_pricing` exists to prevent.

## Prompts are the main behavioural lever

`PLANNER_SYSTEM_PROMPT` (`core/planner.py`) and `WORKER_INSTRUCTIONS`
(`core/worker.py`) drive output depth far more than any scheduler knob. They
are deliberately specific about how a task description must be written —
objective, substance, boundaries, deliverable, quality bar — because a worker
sees nothing but that description. Both are overridable per-swarm via
`build_swarm(planner_system_prompt=..., worker_system_prompt=...)`, so change
them there in experiments rather than editing defaults.

## Planner tool calls are the fragile part

The planner drives the tree through four plain callables in
`make_planner_tools`. Two things bite here:

- `plan_tasks` takes `subtasks` as either a list of objects **or** a JSON
  string. Models emit both; accepting only one shape makes runs die on
  argument validation. Keep any new tool argument equally forgiving.
- `subtasks` may also arrive **missing entirely**. That is not a modelling
  mistake: a tool call that runs past the model's output ceiling comes back
  with its unfinished argument dropped (Anthropic returns the partial input
  object) or unterminated (streaming). The framework then reports "missing 1
  required positional argument", which tells the planner nothing, so it
  retries the same oversized call and the run dies. Hence the default value
  and `_TRUNCATED_ARGS_HINT`, which names the cause and asks for smaller
  batches — and the prompt's cap of 8 subtasks per call. Several `plan_tasks`
  calls into the same parent append siblings, and a later batch can depend on
  ids an earlier one returned.
- `finish(from_task_id=...)` is the door the deliverable goes through, so it
  checks `TRUNCATED_MARKER` on the source task: promoting a cut-off answer
  ships a report that stops mid-sentence, which is the exact outcome the
  marker exists to prevent. It pushes back **once** rather than refusing —
  after a re-plan the planner may legitimately decide a long cut-off answer
  beats the alternative, and a hard refusal would leave a turn-starved run
  with nothing but leaf summaries. The second call on the same id goes
  through and is recorded as `finish_truncated`.
- `finish` refuses while any leaf is pending or in progress, and says why.
  Without that guard a planner that calls `plan_tasks` then `finish` in the
  *same* turn ends the run before a single worker is dispatched — the
  scheduler breaks as soon as `state.finished` is set — so the plan never
  executes and the "answer" is the planner's own guess. The symptom is a run
  that reports one planner turn and zero tasks done. Any new terminal tool
  needs the same check.
- Pydantic AI retries a failed tool-argument validation once and then
  *raises*, which used to abort the whole run. The adapter now passes
  `retries=3`, and `Swarm.arun` catches a failed planner turn, feeds the
  error back as a note, and only gives up after `max_planner_errors`
  consecutive failures. Work already completed must always reach the caller
  — never let a planner-side exception escape `arun`.

## Cost accounting

The core `Swarm` bills `0.0` unless pricing is passed (`planner_pricing` /
`worker_pricing`, $ per million tokens) — that stays true, the library never
guesses a price. The CLI fills the gap: `MODEL_PRICING` in `cli.py` holds
Anthropic list prices and `_model_pricing()` matches a model string against
it (substring, longest key first, so provider prefixes and dated ids
resolve), so `--max-cost` — default `$1.5`, `0` to disable — is enforceable
with no flags on the default models. `--planner-price` / `--worker-price`
override the table; keep the table in sync with
<https://platform.claude.com/docs/en/about-claude/pricing> and prefer the
standard rate over a temporary introductory one, so a run is never
under-billed. Separately, token usage comes from the
provider: with pydantic-ai 1.66 + OpenAI, `usage.input_tokens` is `0`, so cost
tracking is only reliable on Anthropic today. `_cost_summary` in `cli.py`
distinguishes "no list price known" from "provider reported no usage" — keep that
distinction, it is the difference between a config mistake and an upstream
gap.

## Reports (`report.py`)

Worker output is Markdown, so it is parsed once into a shared block model
(`parse_markdown` → `Block`/`Span`) and rendered per format. Add a format by
writing one renderer against that model and registering it in `_WRITERS`;
don't re-parse Markdown per format.

Pipe tables are a block kind of their own (`Block.rows`), recognised only
when the line after the header is a `|---|` separator — otherwise a
sentence containing a pipe would swallow everything under it. A new
renderer has to handle them: they used to fall through to a paragraph, and
a worker's comparison table reached the PDF as a line of pipes. The PDF
hands them to fpdf2's own `table()` (it wraps cells and repeats the header
across page breaks); the deck flattens each row to one bullet, because a
grid cannot own a slide and stay readable.

fpdf2 gotcha: `multi_cell` leaves the cursor at the *right* edge of the cell,
so the next full-width call gets zero width and raises "Not enough horizontal
space". Always go through the local `block_cell` helper.

The PDF needs a Unicode TTF for accented text; `_FONT_CANDIDATES` probes
per-OS paths and falls back to core Helvetica plus a transliteration table.

## Viz

`viz/page.py` is a single HTML string (keeps packaging dependency-free).
Cytoscape.js **and dagre** come from a CDN — the *browser* fetches them, the
myriapod process makes no outbound request. dagre is optional by design:
`HAS_DAGRE` gates the layered layout and falls back to `breadthfirst`, so a
blocked CDN must never break the page. When editing the page, check the JS
with `node --check` after extracting it; there is no bundler and no linter
on it.

To *look* at a change without spending a run: the page holds an SSE
connection open forever, so a headless screenshot never settles. Stub
`window.EventSource` with a fake that emits one canned snapshot, write that
to a temp HTML, and screenshot it with
`Google Chrome --headless=new --virtual-time-budget=9000 --screenshot=...`.
The most faithful version of that harness drives a real offline swarm
(`testing.py` doubles), then feeds `_node_payload(...)` + `tree.planner_turns`
into the stub — same payload the live server sends, no SSE.
Render both a small graph (cards, layered layout) and a 100+ node one (dots,
radial) — they exercise different code paths.

Node labels below the card threshold are **HTML overlays**
(`cytoscape-node-html-label`), because a canvas label is one colour for the
whole string and the cards colour four fields separately. Consequences to
respect: the canvas label must be silenced (`node.htmllabel` rule, declared
last so it wins), and CSS classes like `.faded` do not reach a div — fading
rides in the node's `dim` **data**, which is what the extension re-renders
on. Both the extension and dagre are optional; `HAS_HTML_LABEL` /
`HAS_DAGRE` gate them and must be read *after* `cy` exists (reading
`cy.nodeHtmlLabel` at module top is a temporal-dead-zone crash that blanks
the whole page — it happened).

**The graph holds agents only — a planner turn is a rank, not a node.**
Synthetic `turn-N` nodes existed and were removed: they read as boxes that
are not agents. Every task still carries the `turn` that created it
(`_node_payload`), and a top-level task's hierarchy edge asks dagre for
`minlen = turn` ranks, so wave 2 lands in the column right of wave 1 with
nothing drawn between them. Consequences if you touch it: the edge is
created once and never updated, so `minlen` must be right at creation time;
`layoutOptions` must keep passing `minLen` to dagre or the waves collapse
into one fan; and `maxTurn() > 1` is what forces `rankDir: LR`, since a
multi-wave run *is* a left-to-right sequence. `TaskTree.planner_turns` is
still recorded (reports, post-mortem) but no longer sent on the viz wire —
nothing on the page consumed it once the nodes went away.

`record_planner_turn` books the planner's spend on the **root node**, which
is why `stats()` returns `cost` (total), `workers_cost`
and `planner_cost`: the collector must read `workers_cost`, or the planner
is counted twice. `mark_planner_thinking` flips the root to `in_progress`
and names the model *at the start* of a turn — otherwise a wide plan leaves
the graph looking idle for minutes. Token counts cannot follow: the provider
reports usage when the call returns.

`finish()` lands as a tool call *during* a turn, so `state.finished` is
usually true while the turn is still in flight. The loop waits
`finish_grace` for it rather than cancelling, or the run throws away the
tokens that turn just spent and under-reports its own cost —
`test_root_carries_the_planner_identity_and_spend` pins that down.

Three view modes, all automatic and all overridable by the buttons: cards
below 60 tasks then dots, tree layout below ~10 siblings then radial, and
top-down ranks below 5 siblings then left-to-right. Those thresholds are
about screen shape, not taste — a wide fan-out laid out top-down is an
unreadable strip, which is the whole reason the radial mode exists.

Deltas can carry a node whose dependency sibling has not been added yet, so
edges go through a pending queue — keep any new element handling
order-independent.

## Progress line (`progress.py`)

Without `--viz` a run is silent, and a planner turn can take minutes before
the first worker starts — a terminal printing nothing then is
indistinguishable from a hung one. `status_text` therefore always moves (a
spinner and a rotating verb) and always names a phase, and the phase comes
from the tree: `planner_turns[-1]["running"]` is the *only* signal that a
turn is in flight, since tokens are reported when the call returns. The CLI
and the examples both go through `arun_with_progress`, so there is one
display to fix rather than two. It paints nothing when stderr is not a
terminal — a spinner repainted into a log file is noise.

## Examples

`examples/` is self-contained and numbered: 01-02 run offline (no key, no
cost), 03-06 make small real runs under a `max_cost` set in the file itself.
This repository is public and they are the first thing a reader opens, so
keep them short, keep them honest about what they spend, and keep them
working. A separate `internal-examples/` exists on some checkouts — it is
gitignored, carries Gwenlake-internal context, and has its own `CLAUDE.md`;
nothing in `src/` may depend on it.

## Testing

Everything is offline. `testing.py` provides `ScriptedPlanner` (a planner
double executing a scripted list of tool calls), `SimWorker` (sleeps, then
returns a summary block) and `SimReviewer` (rejects given task ids once,
then passes), plus `Recorder` for max-parallelism assertions. Use those
rather than mocking the adapters. `SimReviewer`'s counters are module-level
so several instances agree on "the first answer only" — call
`reset_review_counts()` between tests.

Watch for tests that assert an order which id-sorting would produce anyway:
two critical-path tests were written that way first and proved nothing. Put
the chain head at a *later* id than the independent tasks.
