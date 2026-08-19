"""What a worker actually receives from its dependencies.

The scaling claim of the project is that context stays bounded. It did not:
a synthesis task declaring twenty dependencies got the concatenation of
twenty full answers, so the context grew with the fan-out — exactly the
thing the design says it avoids.
"""

from myriapod.core.task_tree import TaskTree
from myriapod.core.worker import _fair_share, build_worker_input


def tree_with_deps(sizes: list[int]) -> TaskTree:
    """A fan-in: one synthesis task depending on N outputs of given sizes.

    Each output ends with ``[end-i]`` so a test can tell "quoted whole" from
    "cut" by looking for the tail rather than by counting filler.
    """
    tree = TaskTree("goal")
    subtasks = [{"description": f"part {i}"} for i in range(len(sizes))]
    subtasks.append(
        {
            "description": "synthesis",
            "depends_on": list(range(1, len(sizes) + 1)),
        }
    )
    tree.decompose("root", subtasks)
    for i, size in enumerate(sizes, start=1):
        tail = f"[end-{i}]"
        tree.mark_done(str(i), f"digest of {i}", "x" * (size - len(tail)) + tail)
    return tree


def whole(context: str, ids: range | list[int]) -> int:
    """How many of those dependencies reached the worker uncut."""
    return sum(1 for i in ids if f"[end-{i}]" in context)


def test_a_small_fan_in_passes_through_untouched():
    tree = tree_with_deps([100, 100])
    _, context = build_worker_input(tree, tree.get("3"), context_chars=10_000)
    assert "truncated" not in context
    assert whole(context, [1, 2]) == 2


def test_a_wide_fan_in_is_capped_whatever_the_number_of_dependencies():
    ten = tree_with_deps([20_000] * 10)
    forty = tree_with_deps([20_000] * 40)
    _, small = build_worker_input(ten, ten.get("11"), context_chars=50_000)
    _, large = build_worker_input(forty, forty.get("41"), context_chars=50_000)
    # Boilerplate scales with the dependency count; the payload is what must
    # not. Forty dependencies cost barely more than ten.
    assert len(small) < 60_000
    assert len(large) < 60_000


def test_too_thin_a_slice_becomes_a_digest_instead_of_a_misleading_intro():
    """Forty ways to share 50k is 1250 characters each — an introduction.

    A worker handed forty introductions believes it has read forty reports.
    Below the floor, the digest is both shorter and more honest.
    """
    tree = tree_with_deps([20_000] * 40)
    _, context = build_worker_input(tree, tree.get("41"), context_chars=50_000)
    assert context.count("digest only") == 40
    assert "excerpt: first" not in context
    # And the budget freed by digesting is not spent on nothing.
    assert len(context) < 20_000


def test_the_floor_lifts_the_dependencies_that_can_still_be_quoted():
    """A mixed fan-in keeps quoting whoever fits, at any width.

    Thirty short dependencies pass whole and the two long ones share what is
    left — 10k characters each, well clear of the floor, so they are quoted
    rather than digested.
    """
    tree = tree_with_deps([20_000] * 2 + [800] * 30)
    _, context = build_worker_input(tree, tree.get("33"), context_chars=45_000)
    assert "digest only" not in context
    assert whole(context, range(3, 33)) == 30  # the short ones, untouched
    assert context.count("excerpt: first 10") == 2  # the long ones, quoted


def test_a_trimmed_dependency_carries_its_digest_and_says_what_is_missing():
    tree = tree_with_deps([100_000])
    _, context = build_worker_input(tree, tree.get("2"), context_chars=10_000)
    assert "digest of 1" in context
    assert "characters truncated" in context
    assert "do not invent" in context.lower()
    assert len(context) < 11_000


def test_a_small_dependency_is_not_starved_by_a_large_one():
    tree = tree_with_deps([500, 100_000])
    _, context = build_worker_input(tree, tree.get("3"), context_chars=20_000)
    # The 500-char dependency fits whole; the ~9.5k it does not use goes to
    # the big one rather than being wasted on padding it cannot fill.
    assert whole(context, [1]) == 1
    assert len(context) > 19_000


def test_fair_share_gives_the_slack_of_the_small_to_the_large():
    assert _fair_share([10, 10, 10], 100) == [10, 10, 10]  # everyone fits
    assert _fair_share([10, 1000], 100) == [10, 90]  # spillover
    assert sum(_fair_share([1000, 1000, 1000], 90)) == 90  # exact split
    assert _fair_share([1000], 0) == [0]  # a zero budget terminates
