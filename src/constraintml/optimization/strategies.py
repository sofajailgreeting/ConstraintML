"""Execution strategy types shared between the planner and execution backends."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# int8/fp8 are named here for API completeness but are never selected by the v1
# planner or implemented by the v1 PyTorch backend - only fp32 <-> fp16 is relevant.
PRECISION_LEVELS = ["fp32", "fp16", "int8", "fp8"]


class StrategyAction(str, Enum):
    NO_CHANGE = "no_change"
    REDUCE_PRECISION = "reduce_precision"
    ENABLE_CHECKPOINTING = "enable_checkpointing"
    INCREASE_BATCH_SIZE = "increase_batch_size"
    INCREASE_GRAD_ACCUMULATION = "increase_grad_accumulation"
    EARLY_STOP = "early_stop"


@dataclass(frozen=True)
class ExecutionStrategy:
    precision: str = "fp32"
    batch_size: int = 0  # 0 == "use loader default"; set by the trainer at init
    grad_accumulation_steps: int = 1
    activation_checkpointing: bool = False


@dataclass(frozen=True)
class PlannerDecision:
    action: StrategyAction
    reason: str
    resulting_strategy: ExecutionStrategy
