from playbook.data.loader import DataLoader
from playbook.features.store import FeatureStore
from playbook.models.win_probability import WinProbabilityModel


def test_win_probability_model_training_and_prediction():
    df = DataLoader.generate_synthetic_telemetry(n_games=200, seed=42)
    team_metrics = FeatureStore.compute_team_metrics(df)

    model = WinProbabilityModel()
    model.fit(df, team_metrics)
    assert model.is_fitted is True

    # Avaliação
    df_eval = DataLoader.generate_synthetic_telemetry(n_games=60, seed=123)
    metrics = model.evaluate(df_eval, team_metrics)
    assert "acuracia" in metrics
    assert "roc_auc" in metrics
    assert "brier_score" in metrics
    assert 0.0 <= metrics["acuracia"] <= 1.0

    # Predição
    pred = model.predict_matchup(
        time_casa="Boston Celtics",
        time_visitante="Los Angeles Lakers",
        team_metrics=team_metrics,
    )
    assert 0.0 <= pred.probabilidade_vitoria_casa <= 1.0
    assert pred.classe_vencedora_prevista in ["Boston Celtics", "Los Angeles Lakers"]
