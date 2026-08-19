"""myriapod — planner/worker agent swarms over a dynamic task tree.

A frontier model plans; fleets of cheap models execute — in parallel,
with strict context isolation. © 2026 Gwenlake.
"""

from myriapod.core.protocol import AgentLike, RunOutcome
from myriapod.core.scheduler import Swarm, SwarmCosts, SwarmResult
from myriapod.core.task_tree import TaskNode, TaskStatus, TaskTree, TaskTreeError

__version__ = "0.2.0"

__all__ = [
    "AgentLike",
    "RunOutcome",
    "Swarm",
    "SwarmCosts",
    "SwarmResult",
    "TaskNode",
    "TaskStatus",
    "TaskTree",
    "TaskTreeError",
    "__version__",
]
