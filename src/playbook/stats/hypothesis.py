
import numpy as np
import pandas as pd
from scipy import stats

from playbook.domain.models import HypothesisTestResult


class HypothesisTester:
    """Motor de inferência e validação de hipóteses estatísticas."""

    @staticmethod
    def run_two_sample_ttest(
        coluna_a: str,
        coluna_b: str,
        df: pd.DataFrame,
        nivel_significancia: float = 0.05,
    ) -> HypothesisTestResult:
        """
        Executa o Teste T de Student para duas amostras independentes
        avaliando se as médias diferem significativamente.
        """
        dados_a = df[coluna_a].dropna()
        dados_b = df[coluna_b].dropna()

        t_stat, p_valor = stats.ttest_ind(dados_a, dados_b)
        t_stat = float(t_stat)
        p_valor = float(p_valor)

        rejeita = p_valor < nivel_significancia

        if rejeita:
            conclusao = (
                f"Com p-valor ({p_valor:.4e}) < {nivel_significancia}, rejeita-se a hipótese nula. "
                f"Há evidência estatística significativa de diferença de médias entre {coluna_a} e {coluna_b}."
            )
        else:
            conclusao = (
                f"Com p-valor ({p_valor:.4e}) >= {nivel_significancia}, não se rejeita a hipótese nula. "
                f"Não há evidência estatística suficiente para afirmar que as médias diferem."
            )

        return HypothesisTestResult(
            variavel_a=coluna_a,
            variavel_b=coluna_b,
            t_stat=t_stat,
            p_value=p_valor,
            nivel_significancia=nivel_significancia,
            rejeita_hipotese_nula=rejeita,
            conclusao=conclusao,
        )

    @staticmethod
    def compute_confidence_interval(
        series: pd.Series,
        confidence: float = 0.95,
    ) -> tuple[float, float, float]:
        """Calcula a média amostral e os limites inferior/superior do intervalo de confiança."""
        clean = series.dropna()
        n = len(clean)
        if n < 2:
            val = float(clean.iloc[0]) if n == 1 else 0.0
            return val, val, val

        mean = float(np.mean(clean))
        sem = float(stats.sem(clean))
        h = float(sem * stats.t.ppf((1 + confidence) / 2.0, n - 1))
        return mean, mean - h, mean + h

    @staticmethod
    def sample_data(df: pd.DataFrame, fraction: float = 0.30, seed: int = 42) -> pd.DataFrame:
        """Extrai subamostra determinística para validação rápida de hipóteses."""
        return df.sample(frac=fraction, random_state=seed)
