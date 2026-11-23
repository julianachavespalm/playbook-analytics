import os

from playbook.data.loader import DataLoader


def test_synthetic_telemetry_generation():
    df = DataLoader.generate_synthetic_telemetry(n_games=50, seed=123)
    assert len(df) == 50
    assert "time_casa" in df.columns
    assert "time_visitante" in df.columns
    assert "pontos_casa" in df.columns
    assert "pontos_visitante" in df.columns
    assert (df["pontos_casa"] >= 60).all()


def test_parquet_save_and_load(tmp_path):
    df = DataLoader.generate_synthetic_telemetry(n_games=20, seed=99)
    out_file = str(tmp_path / "test.parquet")
    DataLoader.save_to_parquet(df, out_file)
    assert os.path.exists(out_file)

    loaded = DataLoader.load_from_parquet(out_file)
    assert len(loaded) == 20
    assert list(loaded.columns) == list(df.columns)


def test_duckdb_query():
    df = DataLoader.generate_synthetic_telemetry(n_games=30, seed=77)
    res = DataLoader.query_with_duckdb(
        "SELECT count(*) as total, avg(pontos_casa) as media_casa FROM games",
        games=df,
    )
    assert res.iloc[0]["total"] == 30
    assert res.iloc[0]["media_casa"] > 80
