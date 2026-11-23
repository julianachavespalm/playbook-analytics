import pandas as pd

from playbook.stats.hypothesis import HypothesisTester


def test_hypothesis_tester_significant_difference():
    # Cria amostra com clara diferença entre mandante e visitante
    df = pd.DataFrame({
        "pontos_casa": [115, 118, 120, 114, 116, 122, 119, 117, 121, 115],
        "pontos_visitante": [95, 98, 96, 92, 99, 94, 97, 95, 93, 96],
    })
    result = HypothesisTester.run_two_sample_ttest("pontos_casa", "pontos_visitante", df)
    assert result.rejeita_hipotese_nula is True
    assert result.p_value < 0.05
    assert result.t_stat > 0
    assert "rejeita-se a hipótese nula" in result.conclusao


def test_hypothesis_tester_non_significant_difference():
    # Amostras com médias praticamente iguais
    df = pd.DataFrame({
        "pts_a": [100, 101, 99, 102, 98, 100, 101],
        "pts_b": [100, 99, 101, 100, 102, 99, 100],
    })
    result = HypothesisTester.run_two_sample_ttest("pts_a", "pts_b", df)
    assert result.rejeita_hipotese_nula is False
    assert result.p_value > 0.05
    assert "não se rejeita a hipótese nula" in result.conclusao


def test_confidence_interval():
    series = pd.Series([100.0, 102.0, 98.0, 104.0, 96.0])
    mean, lower, upper = HypothesisTester.compute_confidence_interval(series, confidence=0.95)
    assert mean == 100.0
    assert lower < mean < upper

    # Caso com 1 elemento
    m1, l1, u1 = HypothesisTester.compute_confidence_interval(pd.Series([50.0]))
    assert m1 == l1 == u1 == 50.0


def test_sample_data():
    df = pd.DataFrame({"x": range(100)})
    sampled = HypothesisTester.sample_data(df, fraction=0.30, seed=42)
    assert len(sampled) == 30
