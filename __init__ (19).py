"""
ADAPT-X: adaptive continual learning for tabular supervised tasks under
dynamic distribution shift, with bounded memory.

Public components (each is a real, separately tested module):
    ContinualLearner           src/continual_learner.py     LEARN -> DETECT -> ADAPT -> RETAIN
    DistributionShiftDetector  src/shift_detection.py       bounded-reference drift tests
    AdaptationController       src/adaptation_controller.py drift report -> update plan
    AdaptiveModelState         src/model_state.py           persistent stage library + combiner
    BoundedReplayMemory        src/replay_memory.py         <= K labelled rows
    StreamingFeatureEncoder    src/preprocessing.py         frozen schema, leak-free, bounded
    SequentialEvaluator        src/sequential_evaluation.py test-then-train protocol
"""

from src.adaptation_controller import AdaptationController, UpdatePlan
from src.config import ADAPTXConfig, load_config
from src.continual_learner import ContinualLearner
from src.model_state import AdaptiveModelState
from src.preprocessing import StreamingFeatureEncoder
from src.replay_memory import BoundedReplayMemory
from src.sequential_evaluation import SequentialEvaluator
from src.shift_detection import DistributionShiftDetector, ShiftReport
from src.task import TaskSpec, infer_task_spec

__all__ = [
    "ADAPTXConfig", "load_config", "ContinualLearner", "DistributionShiftDetector", "ShiftReport",
    "AdaptationController", "UpdatePlan", "AdaptiveModelState", "BoundedReplayMemory",
    "StreamingFeatureEncoder", "SequentialEvaluator", "TaskSpec", "infer_task_spec",
]
