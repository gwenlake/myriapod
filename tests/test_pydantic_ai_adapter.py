"""End-to-end wiring check of the Pydantic AI adapter, offline.

Uses Pydantic AI's TestModel ('test' model string): it calls the planner
tools with synthesized arguments and returns canned text. We only assert
that the whole pipeline (tool registration, RunOutcome extraction,
scheduler loop) terminates cleanly — behaviour is meaningless with a
TestModel, wiring is not.
"""

import pytest

from myriapod.adapters.pydantic_ai import build_swarm

pytestmark = pytest.mark.asyncio


async def test_adapter_wiring_with_testmodel():
    swarm = build_swarm(
        planner_model="test",
        worker_model="test",
        max_planner_turns=3,
        worker_timeout=10,
    )
    result = await swarm.arun("wiring check")
    # TestModel behaviour is arbitrary; the contract is: no crash, a result.
    assert result.reason in {
        "finished", "planner_stalled", "max_planner_turns", "max_tasks"
    }
    assert result.duration < 30
