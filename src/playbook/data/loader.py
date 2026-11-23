import os
import sqlite3

import duckdb
import pandas as pd

SQL_CANONICA_JOGADORES = """
SELECT g.game_id, g.team_name_home AS time_casa, g.team_name_away AS time_visitante,
       g.pts_home AS pontos_casa, g.pts_away AS pontos_visitante,
       g.reb_home AS rebotes_casa, g.reb_away AS rebotes_visitante,
       g.ast_home AS assistencias_casa, g.ast_away AS assistencias_visitante,
       cpi.display_first_last AS nome_jogador, cpi.birthdate AS data_nascimento,
       p.is_active AS jogador_ativo,
       CASE
           WHEN pbp.eventmsgtype = 1 THEN 'Pontuação'
           WHEN pbp.eventmsgtype = 7 THEN 'Rebote'
           WHEN pbp.eventmsgtype = 9 THEN 'Assistência'
           WHEN pbp.eventmsgtype = 2 THEN 'Bloqueio'
           WHEN pbp.eventmsgtype = 3 THEN 'Lance Livre'
           WHEN pbp.eventmsgtype = 5 THEN 'Perda de Bola'
           WHEN pbp.eventmsgtype = 6 THEN 'Falta'
           WHEN pbp.eventmsgtype = 8 THEN 'Substituição'
           WHEN pbp.eventmsgtype = 10 THEN 'Salto-bola'
           WHEN pbp.eventmsgtype = 11 THEN 'Expulsão'
           WHEN pbp.eventmsgtype = 12 THEN 'Início de Período'
           WHEN pbp.eventmsgtype = 13 THEN 'Fim de Período'
           WHEN pbp.eventmsgtype = 18 THEN 'Revisão de Suporte'
           ELSE 'Outro Evento'
       END AS tipo_evento,
       pbp.score AS pontuacao
FROM game g
JOIN play_by_play pbp ON g.game_id = pbp.game_id
JOIN common_player_info cpi ON pbp.player1_id = cpi.person_id
JOIN player p ON pbp.player1_id = p.id
WHERE pbp.eventmsgtype IN (1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18);
"""


class DataLoader:
    """Gerenciador de carregamento e ingestão de dados para telemetria de basquete."""

    @staticmethod
    def load_from_sqlite(sqlite_path: str, custom_query: str | None = None) -> pd.DataFrame:
        """Carrega dados a partir de um banco SQLite executando a query canônica."""
        if not os.path.exists(sqlite_path):
            raise FileNotFoundError(f"Banco de dados SQLite não encontrado: {sqlite_path}")

        query = custom_query or SQL_CANONICA_JOGADORES
        with sqlite3.connect(sqlite_path) as conn:
            return pd.read_sql_query(query, conn)

    @staticmethod
    def load_from_parquet(parquet_path: str) -> pd.DataFrame:
        """Carrega telemetria diretamente de arquivo colunar Parquet usando DuckDB/Pandas."""
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Arquivo Parquet não encontrado: {parquet_path}")
        return pd.read_parquet(parquet_path)

    @staticmethod
    def save_to_parquet(df: pd.DataFrame, output_path: str) -> None:
        """Exporta DataFrame processado para o formato colunar Parquet otimizado."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False, compression="snappy")

    @staticmethod
    def query_with_duckdb(query: str, **tables: pd.DataFrame) -> pd.DataFrame:
        """Executa consultas analíticas in-memory em DataFrames via DuckDB."""
        con = duckdb.connect(database=":memory:")
        for name, table in tables.items():
            con.register(name, table)
        return con.execute(query).df()

    @staticmethod
    def generate_synthetic_telemetry(n_games: int = 250, seed: int = 42) -> pd.DataFrame:
        """
        Gera telemetria sintética calibrada com distribuições realistas da NBA
        (mando de campo com média +3.2 pontos e taxa de vitória de mandante de ~59%).
        """
        import numpy as np

        np.random.seed(seed)
        teams = [
            "Boston Celtics", "Los Angeles Lakers", "Golden State Warriors",
            "Miami Heat", "Denver Nuggets", "Milwaukee Bucks", "Dallas Mavericks",
            "Philadelphia 76ers"
        ]

        records = []
        for i in range(n_games):
            home_idx = np.random.randint(0, len(teams))
            away_idx = (home_idx + np.random.randint(1, len(teams))) % len(teams)
            team_home = teams[home_idx]
            team_away = teams[away_idx]

            # Mandante com vantagem estatística modelada
            pts_home = int(np.random.normal(112.5, 11.0))
            pts_away = int(np.random.normal(108.5, 11.0))
            if pts_home == pts_away:
                pts_home += 1  # No basquete não há empates

            reb_home = int(np.random.normal(44.0, 5.0))
            reb_away = int(np.random.normal(41.5, 5.0))
            ast_home = int(np.random.normal(26.0, 4.0))
            ast_away = int(np.random.normal(24.0, 4.0))

            records.append({
                "game_id": f"00224{i:04d}",
                "time_casa": team_home,
                "time_visitante": team_away,
                "pontos_casa": max(60, pts_home),
                "pontos_visitante": max(60, pts_away),
                "rebotes_casa": max(20, reb_home),
                "rebotes_visitante": max(20, reb_away),
                "assistencias_casa": max(10, ast_home),
                "assistencias_visitante": max(10, ast_away),
                "ano": 2024
            })

        return pd.DataFrame(records)
