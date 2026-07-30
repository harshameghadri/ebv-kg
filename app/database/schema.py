"""Database schema initialization and utilities using psycopg."""

from pathlib import Path
from typing import Any

SCHEMA_SQL_PATH = Path(__file__).parent / "schema.sql"


def get_schema_sql() -> str:
    """Read and return the raw PostgreSQL schema SQL script."""
    if not SCHEMA_SQL_PATH.exists():
        raise FileNotFoundError(f"Schema SQL file not found at {SCHEMA_SQL_PATH}")
    return SCHEMA_SQL_PATH.read_text(encoding="utf-8")


def init_db_schema(conn: Any) -> None:
    """Execute the PostgreSQL database schema SQL script against a psycopg connection.

    Idempotent operation using CREATE TABLE IF NOT EXISTS and CREATE INDEX IF NOT EXISTS.

    Args:
        conn: A psycopg Connection (or compatible connection object with execute/commit).
    """
    sql_script = get_schema_sql()
    conn.execute(sql_script)
    if hasattr(conn, "commit"):
        conn.commit()
