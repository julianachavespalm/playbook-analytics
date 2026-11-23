from typing import Any

import pandas as pd

from playbook.domain.models import HomeAdvantageStats, PerformanceTier


class PerformanceClassifier:
    """Classificador de regras de negócio para categorias de pontuação de atletas."""

    @staticmethod
    def classify_score(pontuacao: float) -> str:
        """Classifica a pontuação individual nas 5 categorias normativas."""
        if pontuacao >= 25:
            return PerformanceTier.ELITE.value
        elif pontuacao >= 20:
            return PerformanceTier.ALTO.value
        elif pontuacao >= 15:
            return PerformanceTier.MEDIO.value
        elif pontuacao >= 10:
            return PerformanceTier.REGULAR.value
        else:
            return PerformanceTier.EM_DESENVOLVIMENTO.value

    @classmethod
    def apply_to_dataframe(cls, df: pd.DataFrame, score_column: str = "pontuacao_jogador") -> pd.DataFrame:
        """Aplica a classificação à coluna informada gerando a coluna 'categoria'."""
        df_copy = df.copy()
        df_copy["categoria"] = df_copy[score_column].apply(cls.classify_score)
        return df_copy


class FeatureStore:
    """Motor de engenharia e armazenamento de features de partidas e times."""

    @staticmethod
    def enrich_games(df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriquece o DataFrame de jogos com features de diferencial,
        vitória de mandante e métricas combinadas.
        """
        df_enriched = df.copy()
        df_enriched["vitoria_casa"] = df_enriched["pontos_casa"] > df_enriched["pontos_visitante"]
        df_enriched["diferenca_pontos"] = df_enriched["pontos_casa"] - df_enriched["pontos_visitante"]
        df_enriched["total_pontos"] = df_enriched["pontos_casa"] + df_enriched["pontos_visitante"]
        return df_enriched

    @staticmethod
    def compute_home_advantage_stats(df_enriched: pd.DataFrame) -> tuple[HomeAdvantageStats, pd.DataFrame]:
        """Calcula as estatísticas descritivas completas da vantagem do mandante."""
        if "diferenca_pontos" not in df_enriched.columns:
            df_enriched = FeatureStore.enrich_games(df_enriched)

        diff = df_enriched["diferenca_pontos"]
        q = diff.quantile([0.25, 0.5, 0.75])

        stats = HomeAdvantageStats(
            mediana_diferenca=float(diff.median()),
            media_diferenca=float(diff.mean()),
            desvio_padrao=float(diff.std()),
            percentual_vitorias_casa=float(df_enriched["vitoria_casa"].mean() * 100),
            maior_diferenca=int(diff.max()),
            menor_diferenca=int(diff.min()),
            quartil_25=float(q[0.25]),
            quartil_50=float(q[0.50]),
            quartil_75=float(q[0.75]),
        )

        # Análise temporal (se coluna 'ano' existir)
        if "ano" in df_enriched.columns:
            annual_stats = df_enriched.groupby("ano").agg(
                taxa_vitoria_casa=("vitoria_casa", "mean"),
                media_diferenca=("diferenca_pontos", "mean"),
                desvio_diferenca=("diferenca_pontos", "std"),
            ).reset_index()
        else:
            annual_stats = pd.DataFrame()

        return stats, annual_stats

    @staticmethod
    def compute_team_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Agrega métricas por time como mandante e como visitante."""
        home = df.groupby("time_casa").agg(
            jogos_casa=("game_id", "count"),
            pts_casa_media=("pontos_casa", "mean"),
            reb_casa_media=("rebotes_casa", "mean"),
            ast_casa_media=("assistencias_casa", "mean"),
            vitorias_casa=("pontos_casa", lambda x: (x > df.loc[x.index, "pontos_visitante"]).sum())
        ).reset_index().rename(columns={"time_casa": "time"})

        home["taxa_vitoria_casa"] = home["vitorias_casa"] / home["jogos_casa"]
        return home

    @staticmethod
    def compute_scoring_probability(scores_series: pd.Series, threshold: float = 20.0) -> dict[str, Any]:
        """Calcula a probabilidade empírica de um atleta pontuar acima de um limiar."""
        total = len(scores_series)
        if total == 0:
            return {"total_amostras": 0, "sucessos": 0, "probabilidade": 0.0, "probabilidade_pct": 0.0}

        acima = (scores_series > threshold).sum()
        prob = float(acima / total)
        return {
            "limiar_pontos": threshold,
            "total_amostras": int(total),
            "sucessos": int(acima),
            "probabilidade": prob,
            "probabilidade_pct": round(prob * 100, 2),
        }
