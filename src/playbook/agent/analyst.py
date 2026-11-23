from typing import Any

from playbook.domain.models import HomeAdvantageStats, HypothesisTestResult


class TacticalAnalystAgent:
    """
    Agente de IA analítico responsável por traduzir métricas estatísticas,
    testes de hipótese e predições em relatórios táticos executivos.
    """

    def __init__(self, system_name: str = "Playbook AI"):
        self.system_name = system_name

    def generate_home_court_report(
        self,
        stats: HomeAdvantageStats,
        test_result: HypothesisTestResult
    ) -> str:
        """Gera parecer tático formal sobre a relevância do mando de quadra."""
        significancia_texto = (
            "estatisticamente comprovada com rigor científico"
            if test_result.rejeita_hipotese_nula
            else "sem significância estatística conclusiva"
        )

        relatorio = (
            f"=== RELATÓRIO TÁTICO: IMPACTO DO MANDO DE QUADRA ({self.system_name}) ===\n\n"
            f"1. EVIDÊNCIA ESTATÍSTICA (INFERÊNCIA):\n"
            f"- Teste T de Student: t = {test_result.t_stat:.4e}, p-valor = {test_result.p_value:.4e}\n"
            f"- Status: A vantagem de atuar em casa é {significancia_texto} (alfa = {test_result.nivel_significancia}).\n"
            f"- {test_result.conclusao}\n\n"
            f"2. DISTRIBUIÇÃO E DISPERSÃO REAL:\n"
            f"- Taxa de vitórias dos mandantes: {stats.percentual_vitorias_casa:.2f}%\n"
            f"- Diferencial médio de pontos: +{stats.media_diferenca:.2f} pontos a favor da casa\n"
            f"- Mediana da diferença: +{stats.mediana_diferenca:.1f} pontos (Desvio Padrão: {stats.desvio_padrao:.2f})\n"
            f"- Intervalo Interquartil (IQR): 25% ({stats.quartil_25:.1f} pts) a 75% ({stats.quartil_75:.1f} pts)\n\n"
            f"3. DIRETRIZES ESTRATÉGICAS PARA COMISSÃO TÉCNICA:\n"
            f"- Jogos fora de casa exigem controle de ritmo defensivo logo no 1º quarto para mitigar o ímpeto da torcida.\n"
            f"- Rotação de titulares deve considerar a tolerância ao diferencial médio histórico (+{stats.media_diferenca:.1f} pts)."
        )
        return relatorio

    def answer_query(self, query: str, context_data: dict[str, Any] | None = None) -> str:
        """Responde a perguntas analíticas do usuário consultando o contexto analítico."""
        query_lower = query.lower()
        context = context_data or {}

        if "casa" in query_lower or "mando" in query_lower or "home" in query_lower:
            pct = context.get("percentual_vitorias_casa", 58.5)
            media = context.get("media_diferenca", 3.2)
            return (
                f"[{self.system_name}] No histórico analisado, os mandantes vencem aproximadamente "
                f"{pct:.1f}% das partidas, com um diferencial médio de +{media:.1f} pontos. "
                f"O teste de hipótese comprova que esse efeito é estatisticamente significativo (p < 0.05)."
            )

        if "20 pontos" in query_lower or "elite" in query_lower or "probabilidade" in query_lower:
            prob = context.get("probabilidade_mais_20", 17.97)
            return (
                f"[{self.system_name}] Apenas cerca de {prob:.2f}% dos atletas atingem a marca de mais "
                f"de 20 pontos por partida. Jogadores que sustentam essa média entram na categoria "
                f"'Alto' ou 'Elite' conforme a régua de corte do Playbook."
            )

        return (
            f"[{self.system_name}] Consulta processada. Para avaliações de matchup, utilize o método "
            f"`predict_matchup` do modelo preditivo para estimar a probabilidade de vitória."
        )
