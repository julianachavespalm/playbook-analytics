import pandas as pd

from playbook.domain.models import PerformanceTier
from playbook.features.store import FeatureStore, PerformanceClassifier


def test_performance_classifier_thresholds():
    assert PerformanceClassifier.classify_score(30) == PerformanceTier.ELITE.value
    assert PerformanceClassifier.classify_score(25) == PerformanceTier.ELITE.value
    assert PerformanceClassifier.classify_score(22) == PerformanceTier.ALTO.value
    assert PerformanceClassifier.classify_score(20) == PerformanceTier.ALTO.value
    assert PerformanceClassifier.classify_score(18) == PerformanceTier.MEDIO.value
    assert PerformanceClassifier.classify_score(15) == PerformanceTier.MEDIO.value
    assert PerformanceClassifier.classify_score(12) == PerformanceTier.REGULAR.value
    assert PerformanceClassifier.classify_score(10) == PerformanceTier.REGULAR.value
    assert PerformanceClassifier.classify_score(8) == PerformanceTier.EM_DESENVOLVIMENTO.value
    assert PerformanceClassifier.classify_score(0) == PerformanceTier.EM_DESENVOLVIMENTO.value


def test_performance_classifier_dataframe():
    df = pd.DataFrame({"pontuacao_jogador": [27, 21, 17, 11, 4]})
    classified = PerformanceClassifier.apply_to_dataframe(df)
    assert list(classified["categoria"]) == [
        PerformanceTier.ELITE.value,
        PerformanceTier.ALTO.value,
        PerformanceTier.MEDIO.value,
        PerformanceTier.REGULAR.value,
        PerformanceTier.EM_DESENVOLVIMENTO.value,
    ]


def test_feature_store_enrichment():
    df = pd.DataFrame({
        "game_id": ["001", "002"],
        "time_casa": ["A", "B"],
        "time_visitante": ["B", "A"],
        "pontos_casa": [110, 95],
        "pontos_visitante": [105, 100],
        "rebotes_casa": [45, 38],
        "rebotes_visitante": [40, 42],
        "assistencias_casa": [25, 20],
        "assistencias_visitante": [22, 24],
    })
    enriched = FeatureStore.enrich_games(df)
    assert list(enriched["vitoria_casa"]) == [True, False]
    assert list(enriched["diferenca_pontos"]) == [5, -5]
    assert list(enriched["total_pontos"]) == [215, 195]


def test_compute_scoring_probability():
    scores = pd.Series([28, 22, 19, 14, 8, 25, 10, 31, 16, 5])
    prob_dict = FeatureStore.compute_scoring_probability(scores, threshold=20.0)
    assert prob_dict["total_amostras"] == 10
    assert prob_dict["sucessos"] == 4  # 28, 22, 25, 31
    assert prob_dict["probabilidade"] == 0.4
    assert prob_dict["probabilidade_pct"] == 40.0


def test_compute_home_advantage_stats():
    df = pd.DataFrame({
        "game_id": ["001", "002", "003"],
        "time_casa": ["A", "B", "A"],
        "time_visitante": ["B", "A", "B"],
        "pontos_casa": [110, 95, 120],
        "pontos_visitante": [105, 100, 110],
        "ano": [2024, 2024, 2024],
    })
    stats, annual = FeatureStore.compute_home_advantage_stats(df)
    assert stats.percentual_vitorias_casa > 50.0
    assert stats.media_diferenca > 0
    assert len(annual) == 1
    assert annual.iloc[0]["ano"] == 2024


def test_compute_team_metrics():
    df = pd.DataFrame({
        "game_id": ["001", "002"],
        "time_casa": ["A", "B"],
        "time_visitante": ["B", "A"],
        "pontos_casa": [110, 95],
        "pontos_visitante": [105, 100],
        "rebotes_casa": [45, 38],
        "rebotes_visitante": [40, 42],
        "assistencias_casa": [25, 20],
        "assistencias_visitante": [22, 24],
    })
    metrics = FeatureStore.compute_team_metrics(df)
    assert len(metrics) == 2
    assert "time" in metrics.columns
    assert "taxa_vitoria_casa" in metrics.columns

