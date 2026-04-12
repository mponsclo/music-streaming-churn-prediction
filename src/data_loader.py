"""
DuckDB-based data loading utilities.

All heavy CSV reads and aggregations run through DuckDB, which streams
large files without loading them fully into memory. Results come back
as pandas DataFrames for visualization and modeling.
"""

from pathlib import Path

import duckdb
import pandas as pd


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return a fresh in-memory DuckDB connection."""
    return duckdb.connect()


def query_csv(path: Path, sql: str, con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """
    Run a SQL query against a CSV file and return a pandas DataFrame.

    The token {path} inside the SQL string will be replaced with
    the actual file path wrapped in read_csv_auto().

    Parameters
    ----------
    path : Path
        Path to the CSV file.
    sql : str
        SQL query. Use ``read_csv_auto('{path}')`` to reference the file,
        or just pass a query that already contains the full path.
    con : DuckDBPyConnection, optional
        Existing connection. A temporary one is created if omitted.

    Example
    -------
    >>> query_csv(TRAIN_PATH, "SELECT * FROM read_csv_auto('{path}') LIMIT 5")
    """
    if con is None:
        con = get_connection()
    resolved = sql.replace("{path}", str(path))
    return con.execute(resolved).fetchdf()


def describe_csv(path: Path, con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """Return column names and inferred types for a CSV file."""
    return query_csv(
        path,
        "DESCRIBE SELECT * FROM read_csv_auto('{path}')",
        con=con,
    )


def count_rows(path: Path, con: duckdb.DuckDBPyConnection | None = None) -> int:
    """Return the row count of a CSV file (fast, streamed)."""
    df = query_csv(
        path,
        "SELECT COUNT(*) AS n FROM read_csv_auto('{path}')",
        con=con,
    )
    return int(df["n"].iloc[0])
