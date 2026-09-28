"""V3 cross-asset research pipeline kept independent from frozen V2."""

from qqq_agents.v3.allocation import ContinuousAllocator
from qqq_agents.v3.config import V3Config, load_v3_config
from qqq_agents.v3.evaluation import V3WalkForwardResult, run_v3_walk_forward
from qqq_agents.v3.features import build_cross_asset_panel, cash_returns_from_yield

__all__ = [
    "ContinuousAllocator",
    "V3Config",
    "V3WalkForwardResult",
    "build_cross_asset_panel",
    "cash_returns_from_yield",
    "load_v3_config",
    "run_v3_walk_forward",
]
