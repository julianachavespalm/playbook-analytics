from enum import Enum

from pydantic import BaseModel, Field


class EventType(str, Enum):
    PONTUACAO = "Pontuação"
    REBOTE = "Rebote"
    ASSISTENCIA = "Assistência"
    BLOQUEIO = "Bloqueio"
    LANCE_LIVRE = "Lance Livre"
    PERDA_DE_BOLA = "Perda de Bola"
    FALTA = "Falta"
    SUBSTITUICAO = "Substituição"
    SALTO_BOLA = "Salto-bola"
    EXPULSAO = "Expulsão"
    INICIO_PERIODO = "Início de Período"
    FIM_PERIODO = "Fim de Período"
    REVISAO_SUPORTE = "Revisão de Suporte"
    OUTRO = "Outro Evento"


class PerformanceTier(str, Enum):
    ELITE = "Elite"
    ALTO = "Alto"
    MEDIO = "Médio"
    REGULAR = "Regular"
    EM_DESENVOLVIMENTO = "Em Desenvolvimento"


class GameRecord(BaseModel):
    game_id: str
    time_casa: str
    time_visitante: str
    pontos_casa: int = Field(ge=0)
    pontos_visitante: int = Field(ge=0)
    rebotes_casa: int = Field(default=0, ge=0)
    rebotes_visitante: int = Field(default=0, ge=0)
    assistencias_casa: int = Field(default=0, ge=0)
    assistencias_visitante: int = Field(default=0, ge=0)
    data_jogo: str | None = None


class HomeAdvantageStats(BaseModel):
    mediana_diferenca: float
    media_diferenca: float
    desvio_padrao: float
    percentual_vitorias_casa: float
    maior_diferenca: int
    menor_diferenca: int
    quartil_25: float
    quartil_50: float
    quartil_75: float


class HypothesisTestResult(BaseModel):
    variavel_a: str
    variavel_b: str
    t_stat: float
    p_value: float
    nivel_significancia: float
    rejeita_hipotese_nula: bool
    conclusao: str


class PredictionResult(BaseModel):
    time_casa: str
    time_visitante: str
    probabilidade_vitoria_casa: float
    margem_estimada_pontos: float
    classe_vencedora_prevista: str
