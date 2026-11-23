
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from playbook.domain.models import PredictionResult


class WinProbabilityModel:
    """
    Modelo de Machine Learning para estimativa de probabilidade de vitória
    e margem de pontos no basquete, integrando efeito de mando de quadra.
    """

    def __init__(self, random_state: int = 42):
        self.model = LogisticRegression(random_state=random_state)
        self.is_fitted = False
        self.feature_names = [
            "diff_pts_historico",
            "diff_reb_historico",
            "diff_ast_historico",
            "fator_casa"
        ]

    def _extract_features(
        self,
        df: pd.DataFrame,
        team_metrics: pd.DataFrame | None = None
    ) -> tuple[np.ndarray, np.ndarray]:

        # Se não foram passadas métricas históricas agregadas, calculamos internamente
        if team_metrics is None:
            team_stats = df.groupby("time_casa").agg(
                pts_mean=("pontos_casa", "mean"),
                reb_mean=("rebotes_casa", "mean"),
                ast_mean=("assistencias_casa", "mean"),
            ).to_dict("index")
        else:
            team_stats = team_metrics.set_index("time").to_dict("index")

        X_list = []
        y_list = []

        for _, row in df.iterrows():
            home = row["time_casa"]
            away = row["time_visitante"]

            home_s = team_stats.get(home, {"pts_mean": 105.0, "reb_mean": 42.0, "ast_mean": 24.0})
            away_s = team_stats.get(away, {"pts_mean": 105.0, "reb_mean": 42.0, "ast_mean": 24.0})

            diff_pts = home_s.get("pts_mean", 105.0) - away_s.get("pts_mean", 105.0)
            diff_reb = home_s.get("reb_mean", 42.0) - away_s.get("reb_mean", 42.0)
            diff_ast = home_s.get("ast_mean", 24.0) - away_s.get("ast_mean", 24.0)
            fator_casa = 1.0  # Vantagem de jogar em casa como feature explícita

            X_list.append([diff_pts, diff_reb, diff_ast, fator_casa])
            y_list.append(1 if row["pontos_casa"] > row["pontos_visitante"] else 0)

        return np.array(X_list), np.array(y_list)

    def fit(self, df_train: pd.DataFrame, team_metrics: pd.DataFrame | None = None) -> "WinProbabilityModel":
        """Treina o estimador de probabilidade de vitória."""
        X, y = self._extract_features(df_train, team_metrics)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def evaluate(self, df_test: pd.DataFrame, team_metrics: pd.DataFrame | None = None) -> dict[str, float]:
        """Avalia métricas de calibração e precisão do modelo preditivo."""
        if not self.is_fitted:
            raise ValueError("O modelo precisa ser treinado antes da avaliação.")

        X, y = self._extract_features(df_test, team_metrics)
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)[:, 1]

        acc = float(accuracy_score(y, preds))
        auc = float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else 0.5
        brier = float(brier_score_loss(y, probs))

        return {
            "acuracia": round(acc, 4),
            "roc_auc": round(auc, 4),
            "brier_score": round(brier, 4),
        }

    def predict_matchup(
        self,
        time_casa: str,
        time_visitante: str,
        team_metrics: pd.DataFrame
    ) -> PredictionResult:
        """Gera inferência probabilística para um confronto específico."""
        if not self.is_fitted:
            raise ValueError("O modelo precisa ser treinado antes de realizar inferências.")

        stats_map = team_metrics.set_index("time").to_dict("index")
        home_s = stats_map.get(time_casa, {"pts_casa_media": 110.0, "reb_casa_media": 43.0, "ast_casa_media": 25.0})
        away_s = stats_map.get(time_visitante, {"pts_casa_media": 108.0, "reb_casa_media": 42.0, "ast_casa_media": 24.0})

        diff_pts = home_s.get("pts_casa_media", 110.0) - away_s.get("pts_casa_media", 108.0)
        diff_reb = home_s.get("reb_casa_media", 43.0) - away_s.get("reb_casa_media", 42.0)
        diff_ast = home_s.get("ast_casa_media", 25.0) - away_s.get("ast_casa_media", 24.0)
        fator_casa = 1.0

        X = np.array([[diff_pts, diff_reb, diff_ast, fator_casa]])
        prob_casa = float(self.model.predict_proba(X)[0, 1])

        # Estimativa heurística da margem de pontos baseada na probabilidade
        margem_estimada = round((prob_casa - 0.5) * 16.0, 1)

        vencedor = time_casa if prob_casa >= 0.5 else time_visitante

        return PredictionResult(
            time_casa=time_casa,
            time_visitante=time_visitante,
            probabilidade_vitoria_casa=round(prob_casa, 4),
            margem_estimada_pontos=margem_estimada,
            classe_vencedora_prevista=vencedor,
        )
