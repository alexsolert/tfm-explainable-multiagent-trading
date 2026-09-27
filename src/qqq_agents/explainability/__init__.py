"""Explicaciones post-hoc para agentes cuantitativos."""

from qqq_agents.explainability.cases import generate_lime_cases, select_representative_cases
from qqq_agents.explainability.local import LimeAgentExplainer, TreeShapAgentExplainer

__all__ = [
    "LimeAgentExplainer",
    "TreeShapAgentExplainer",
    "generate_lime_cases",
    "select_representative_cases",
]
