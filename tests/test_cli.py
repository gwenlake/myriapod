import json
import pytest
import typer
from typer.testing import CliRunner

from myriapod.cli import (
    _planning_note,
    _model_pricing,
    _parse_pricing,
    _print_result,
    app,
    console,
)
from myriapod.core.scheduler import SwarmCosts, SwarmResult

runner = CliRunner()


def _result(content, reason="finished", **costs) -> SwarmResult:
    return SwarmResult(
        content=content,
        finished=content is not None,
        reason=reason,
        tree={},
        costs=SwarmCosts(**costs),
        planner_turns=1,
        tasks_done=0,
        tasks_failed=0,
        duration=0.5,
    )


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    from myriapod import __version__ as v; assert v in result.output


def test_bench_smoke():
    result = runner.invoke(
        app, ["bench", "-n", "40", "-c", "40", "--delay", "0.005", "--jitter", "0"]
    )
    assert result.exit_code == 0, result.output


def test_ask_help_lists_options():
    result = runner.invoke(app, ["ask", "--help"])
    assert result.exit_code == 0
    for opt in ("--planner", "--worker", "--concurrency", "--max-cost", "--web"):
        assert opt in result.output


def test_ask_and_bench_expose_viz_flags():
    for cmd in ("ask", "bench"):
        result = runner.invoke(app, [cmd, "--help"])
        assert result.exit_code == 0
        assert "--viz" in result.output and "--viz-port" in result.output


def test_ask_exposes_pricing_flags():
    result = runner.invoke(app, ["ask", "--help"])
    assert "--planner-price" in result.output and "--worker-price" in result.output


def test_planning_note_carries_the_run_parameters():
    # the note is built from the flags, and never mixed into the goal: the
    # root node and the report title stay the user's own words
    note = _planning_note(None, 8)
    assert "--concurrency 8" in note and "--fanout" not in note
    wide = _planning_note(100, 50)
    assert "--fanout 100" in wide and "--concurrency 50" in wide
    assert "plan_tasks" in wide


def test_model_pricing_defaults():
    # the shipped defaults are billed without any --*-price flag
    assert _model_pricing("anthropic:claude-opus-5") == {"input": 5.0, "output": 25.0}
    assert _model_pricing("anthropic:claude-haiku-4-5") == {"input": 1.0, "output": 5.0}
    # provider prefixes and dated ids resolve to the same entry
    assert _model_pricing("bedrock:anthropic.claude-haiku-4-5-20251001") == {
        "input": 1.0, "output": 5.0,
    }
    # a longer key never loses to a shorter, ambiguous one
    assert _model_pricing("anthropic:claude-opus-4-5") == {"input": 5.0, "output": 25.0}
    assert _model_pricing("openai:gpt-5") is None


def test_parse_pricing():
    assert _parse_pricing("1.25/10", "planner") == {"input": 1.25, "output": 10.0}
    assert _parse_pricing(None, "planner") is None
    with pytest.raises(typer.Exit):
        _parse_pricing("nope", "planner")


def _stderr(result: SwarmResult, **kwargs) -> str:
    with console.capture() as cap:
        _print_result(result, as_json=False, save_tree=None, **kwargs)
    return cap.get()


def test_no_content_is_reported_not_silent():
    out = _stderr(_result(None, reason="planner_stalled"))
    assert "no answer produced" in out
    assert "planner_stalled" in out
    # the hint tells the user what to do about it
    assert "tool calling" in out


def test_cost_line_flags_missing_pricing_and_missing_usage():
    tokens_no_price = _stderr(
        _result("ok", planner_input_tokens=100, planner_output_tokens=50)
    )
    assert "--planner-price" in tokens_no_price

    no_usage = _stderr(_result("ok"))
    assert "cost unknown" in no_usage

    priced = _stderr(
        _result("ok", planner_input_tokens=100, planner_cost=0.25, workers_cost=0.25)
    )
    assert "$0.5000" in priced


def test_resume_rejects_a_tree_it_cannot_read_before_spending_anything(tmp_path):
    """A bad --resume must fail before a fresh run is planned and paid for."""
    broken = tmp_path / "tree.json"
    broken.write_text("not json at all")
    result = runner.invoke(app, ["ask", "q", "--resume", str(broken)])
    assert result.exit_code == 2
    assert "cannot resume" in result.output


def test_resume_reads_a_tree_written_by_save_tree(tmp_path):
    """--save-tree writes it, --resume reads it: the round trip must hold."""
    from myriapod.core.task_tree import TaskTree

    tree = TaskTree("q")
    tree.decompose("root", [{"description": "a"}, {"description": "b"}])
    tree.mark_done("1", "digest", "FULL OUTPUT")
    path = tmp_path / "tree.json"
    path.write_text(tree.to_json())

    restored = TaskTree.from_dict(json.loads(path.read_text()))
    assert restored.get("1").result_full == "FULL OUTPUT"
    assert [n.id for n in restored.ready_tasks()] == ["2"]
