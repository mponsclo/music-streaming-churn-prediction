"""Tests for src/data_loader.py DuckDB helpers."""

from src.data_loader import count_rows, describe_csv, get_connection, query_csv


def test_get_connection_returns_usable_connection():
    con = get_connection()
    result = con.execute("SELECT 1 AS x").fetchone()
    assert result == (1,)


def test_query_csv_returns_expected_rows(tiny_csv):
    df = query_csv(tiny_csv, "SELECT * FROM read_csv_auto('{path}')")
    assert len(df) == 5
    assert set(df.columns) == {"msno", "is_churn", "total_secs"}


def test_query_csv_supports_aggregation(tiny_csv):
    df = query_csv(
        tiny_csv,
        "SELECT SUM(is_churn) AS churners FROM read_csv_auto('{path}')",
    )
    assert int(df["churners"].iloc[0]) == 2


def test_count_rows(tiny_csv):
    assert count_rows(tiny_csv) == 5


def test_describe_csv_lists_columns(tiny_csv):
    df = describe_csv(tiny_csv)
    assert "column_name" in df.columns
    assert set(df["column_name"]) == {"msno", "is_churn", "total_secs"}
