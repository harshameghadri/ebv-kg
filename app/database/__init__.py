"""Database module for EBV Knowledge System."""

from app.database.schema import get_schema_sql, init_db_schema

__all__ = ["get_schema_sql", "init_db_schema"]
