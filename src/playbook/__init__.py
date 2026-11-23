"""
Playbook Analytics
Plataforma de inteligência analítica, engenharia de features e modelagem preditiva para basquete.
"""

from playbook.agent.analyst import TacticalAnalystAgent
from playbook.data.loader import SQL_CANONICA_JOGADORES, DataLoader
from playbook.domain.models import (
    EventType,
    GameRecord,
    HomeAdvantageStats,
    HypothesisTestResult,
    PerformanceTier,
    PredictionResult,
)
from playbook.features.store import FeatureStore, PerformanceClassifier
from playbook.models.win_probability import WinProbabilityModel
from playbook.stats.hypothesis import HypothesisTester

__version__ = "0.1.0"
__author__ = "Juliana Chaves Palm"

__all__ = [
    "SQL_CANONICA_JOGADORES",
    "DataLoader",
    "EventType",
    "FeatureStore",
    "GameRecord",
    "HomeAdvantageStats",
    "HypothesisTestResult",
    "HypothesisTester",
    "PerformanceClassifier",
    "PerformanceTier",
    "PredictionResult",
    "TacticalAnalystAgent",
    "WinProbabilityModel",
]
