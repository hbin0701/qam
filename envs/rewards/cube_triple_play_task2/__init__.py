"""Dense reward models for cube-triple-play-task2.

Available versions:
- v1: Simple cube placement count (0-3 progress)
- v2: Distance-based progress (0-3 progress with sub-progress)
- v3: Detailed stage tracking (0-9 progress, strict cube ordering)
"""

from .dense_v1 import Task2EnvV1, progress_v1, generate_plan_list_v1, determination_v1
from .dense_v2 import Task2EnvV2, progress_v2, generate_plan_list_v2, determination_v2
from .dense_v3 import Task2EnvV3, progress_v3, generate_plan_list_v3, determination_v3
from .wrapper import DenseRewardWrapper

__all__ = [
    # V1
    'Task2EnvV1',
    'progress_v1',
    'generate_plan_list_v1',
    'determination_v1',
    # V2
    'Task2EnvV2',
    'progress_v2',
    'generate_plan_list_v2',
    'determination_v2',
    # V3
    'Task2EnvV3',
    'progress_v3',
    'generate_plan_list_v3',
    'determination_v3',
    # Wrapper
    'DenseRewardWrapper',
]
