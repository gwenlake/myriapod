import threading

import pytest

from myriapod.core.task_tree import TaskStatus, TaskTree, TaskTreeError


def make_tree(**kwargs) -> TaskTree:
    return TaskTree("Write a report", **kwargs)


def test_decompose_assigns_hierarchical_ids():
    tree = make_tree()
    created = tree.decompose("root", [{"description": "A"}, {"description": "B"}])
    assert [n.id for n in created] == ["1", "2"]
    sub = tree.decompose("1", [{"description": "A1"}, {"description": "A2"}])
    assert [n.id for n in sub] == ["1.1", "1.2"]
    # Appending to an existing parent continues the numbering.
    more = tree.decompose("root", [{"description": "C"}])
    assert more[0].id == "3"


def test_dependencies_by_index_and_by_id():
    tree = make_tree()
    tree.decompose(
        "root",
        [
            {"description": "research"},
            {"description": "outline"},
            {"description": "write", "depends_on": [1, 2]},
        ],
    )
    ready = {n.id for n in tree.ready_tasks()}
    assert ready == {"1", "2"}
    tree.decompose("root", [{"description": "review", "depends_on": ["3"]}])
    assert tree.get("4").depends_on == ["3"]


def test_cycle_detection():
    tree = make_tree()
    with pytest.raises(TaskTreeError, match="cycle"):
        tree.decompose(
            "root",
            [
                {"description": "A", "depends_on": [2]},
                {"description": "B", "depends_on": [1]},
            ],
        )
    # Atomicity: nothing was created.
    assert tree.ready_tasks() == []


def test_max_depth_guard():
    tree = make_tree(max_depth=2)
    tree.decompose("root", [{"description": "A"}])
    tree.decompose("1", [{"description": "A1"}])
    with pytest.raises(TaskTreeError, match="Max depth"):
        tree.decompose("1.1", [{"description": "too deep"}])


def test_dependency_gates_ready_tasks_and_unblocks_on_done():
    tree = make_tree()
    tree.decompose(
        "root",
        [
            {"description": "part 1"},
            {"description": "part 2"},
            {"description": "merge", "depends_on": [1, 2]},
        ],
    )
    tree.mark_done("1", "done 1", "full 1")
    assert {n.id for n in tree.ready_tasks()} == {"2"}
    tree.mark_done("2", "done 2", "full 2")
    assert {n.id for n in tree.ready_tasks()} == {"3"}


def test_completion_bubbles_to_root():
    tree = make_tree()
    tree.decompose("root", [{"description": "A"}, {"description": "B"}])
    tree.decompose("2", [{"description": "B1"}, {"description": "B2"}])
    tree.mark_done("1", "s1", "f1")
    tree.mark_done("2.1", "s21", "f21")
    assert not tree.is_complete()
    tree.mark_done("2.2", "s22", "f22")
    assert tree.get("2").status is TaskStatus.DONE
    assert tree.is_complete()
    # Internal node aggregates its children's summaries.
    assert "s21" in tree.get("2").result_summary


def test_skip_satisfies_dependents():
    tree = make_tree()
    tree.decompose(
        "root",
        [{"description": "flaky"}, {"description": "after", "depends_on": [1]}],
    )
    tree.mark_failed("1", "boom")
    assert tree.ready_tasks() == []
    tree.skip_task("1", "unreachable data source")
    assert {n.id for n in tree.ready_tasks()} == {"2"}
    assert "[skipped]" in tree.get("1").result_summary


def test_reset_task_after_failure():
    tree = make_tree()
    tree.decompose("root", [{"description": "A"}])
    tree.mark_in_progress("1", worker="w")
    tree.mark_failed("1", "network error")
    tree.reset_task("1")
    node = tree.get("1")
    assert node.status is TaskStatus.PENDING
    assert node.error is None
    assert node.attempts == 1  # attempts persist across retries


def test_dependency_context_aggregates_internal_deps():
    tree = make_tree()
    tree.decompose(
        "root",
        [{"description": "research"}, {"description": "write", "depends_on": [1]}],
    )
    tree.decompose("1", [{"description": "r1"}, {"description": "r2"}])
    tree.mark_done("1.1", "s11", "FULL-11")
    tree.mark_done("1.2", "s12", "FULL-12")
    context = tree.dependency_context("2")
    assert len(context) == 1
    (title, blob), = context.items()
    assert title.startswith("1 — ")
    assert "FULL-11" in blob and "FULL-12" in blob


def test_render_shows_summaries_never_full_outputs():
    tree = make_tree()
    tree.decompose("root", [{"description": "A"}])
    tree.mark_done("1", "short digest", "VERY LONG RAW OUTPUT " * 50)
    rendered = tree.render()
    assert "short digest" in rendered
    assert "VERY LONG RAW OUTPUT" not in rendered
    assert "[1]" in rendered


def test_serialization_roundtrip():
    tree = make_tree()
    tree.decompose(
        "root",
        [{"description": "A"}, {"description": "B", "depends_on": [1]}],
    )
    tree.mark_done("1", "s", "f", input_tokens=10, output_tokens=5, cost=0.01)
    clone = TaskTree.from_dict(tree.to_dict())
    assert clone.goal == tree.goal
    assert clone.get("1").status is TaskStatus.DONE
    assert clone.get("2").depends_on == ["1"]
    assert clone.stats()["cost"] == pytest.approx(0.01)


def test_thread_safety_smoke():
    tree = make_tree()
    tree.decompose("root", [{"description": f"t{i}"} for i in range(50)])
    ids = [n.id for n in tree.ready_tasks()]

    def work(node_id: str) -> None:
        tree.mark_in_progress(node_id)
        tree.mark_done(node_id, f"done {node_id}", "full", input_tokens=1, cost=0.001)

    threads = [threading.Thread(target=work, args=(i,)) for i in ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert tree.is_complete()
    stats = tree.stats()
    assert stats["by_status"]["done"] == 50
    assert stats["input_tokens"] == 50


def test_reset_reopens_autocompleted_ancestors():
    tree = make_tree()
    tree.decompose("root", [{"description": "A"}])
    tree.decompose("1", [{"description": "A1"}, {"description": "A2"}])
    tree.mark_done("1.1", "s", "f")
    tree.mark_done("1.2", "s", "f")
    assert tree.is_complete()
    tree.reset_task("1.1")
    assert not tree.is_complete()
    assert tree.get("1").status is TaskStatus.IN_PROGRESS
    assert {n.id for n in tree.ready_tasks()} == {"1.1"}
    tree.mark_done("1.1", "s2", "f2")
    assert tree.is_complete()


def test_render_collapses_settled_branches_and_caps_lines():
    tree = make_tree()
    tree.decompose("root", [{"description": "batch"}, {"description": "active"}])
    tree.decompose("1", [{"description": f"t{i}"} for i in range(40)])
    for node in tree.ready_tasks():
        if node.id.startswith("1."):
            tree.mark_done(node.id, f"done {node.id}", "full")
    rendered = tree.render(collapse_done=True)
    assert "(40 tasks done)" in rendered
    assert "[1.17]" not in rendered  # no per-child line in collapsed branches
    assert "· [2] active" in rendered
    assert rendered.strip().endswith("]")  # stats footer present
    full = tree.render(collapse_done=False, max_lines=30)
    assert "lines omitted" in full


def test_large_tree_ready_tasks_is_fast():
    import time as _t

    tree = make_tree(max_depth=2)
    tree.decompose("root", [{"description": f"t{i}"} for i in range(2000)])
    started = _t.perf_counter()
    for node in tree.ready_tasks():
        tree.mark_in_progress(node.id)
        tree.mark_done(node.id, "s", "f")
    elapsed = _t.perf_counter() - started
    assert tree.is_complete()
    assert elapsed < 5.0, f"2000-task drain took {elapsed:.2f}s"


def test_a_settled_container_reopens_when_planned_into():
    """The planner reads a finished batch and adds follow-ups: that means
    decomposing a node whose children have all settled. Refusing it made the
    "plan follow-up tasks" instruction impossible to obey."""
    tree = TaskTree("goal")
    first = tree.decompose("root", [{"description": "a"}, {"description": "b"}])
    for node in first:
        tree.mark_done(node.id, "full", "summary")
    assert tree.is_complete()

    second = tree.decompose("root", [{"description": "c"}])
    assert [n.id for n in second] == ["3"]
    assert not tree.is_complete()                 # the run is live again
    assert tree.get("root").status is TaskStatus.IN_PROGRESS
    assert tree.get("root").result_summary is None  # stale digest dropped

    tree.mark_done("3", "full", "summary")
    assert tree.is_complete()                     # counters still add up


def test_a_finished_leaf_is_not_turned_into_a_container():
    """It owns an output; decomposing it would orphan that. Say so instead."""
    tree = TaskTree("goal")
    (leaf,) = tree.decompose("root", [{"description": "a"}])
    tree.mark_done(leaf.id, "full", "summary")
    with pytest.raises(TaskTreeError, match="finished and has its own output"):
        tree.decompose(leaf.id, [{"description": "follow-up"}])
