from playbook.agent.analyst import TacticalAnalystAgent
from playbook.domain.models import HomeAdvantageStats, HypothesisTestResult


def test_tactical_analyst_agent_report_generation():
    agent = TacticalAnalystAgent(system_name="Playbook Test AI")
    stats = HomeAdvantageStats(
        mediana_diferenca=3.0,
        media_diferenca=3.25,
        desvio_padrao=10.5,
        percentual_vitorias_casa=58.8,
        maior_diferenca=35,
        menor_diferenca=-28,
        quartil_25=-4.0,
        quartil_50=3.0,
        quartil_75=10.0,
    )
    test_res = HypothesisTestResult(
        variavel_a="pontos_casa",
        variavel_b="pontos_visitante",
        t_stat=3.88e2,
        p_value=1.2e-12,
        nivel_significancia=0.05,
        rejeita_hipotese_nula=True,
        conclusao="Diferença significativa comprovada.",
    )

    report = agent.generate_home_court_report(stats, test_res)
    assert "Playbook Test AI" in report
    assert "58.80%" in report
    assert "estatisticamente comprovada" in report


def test_tactical_analyst_agent_qa():
    agent = TacticalAnalystAgent()
    resp_casa = agent.answer_query("Qual a importância do mando de casa?")
    assert "mandantes vencem" in resp_casa

    resp_score = agent.answer_query("Qual a probabilidade de um jogador fazer mais de 20 pontos?")
    assert "20 pontos" in resp_score
