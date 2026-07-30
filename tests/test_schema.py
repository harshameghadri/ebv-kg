"""Unit tests for PostgreSQL database schema definition and initialization."""

import re
from unittest.mock import MagicMock
from pathlib import Path

import pytest
from app.database.schema import get_schema_sql, init_db_schema, SCHEMA_SQL_PATH


def test_schema_sql_file_exists():
    assert SCHEMA_SQL_PATH.exists(), "schema.sql file must exist"
    sql = get_schema_sql()
    assert len(sql.strip()) > 0, "schema.sql must not be empty"


def test_schema_tables_present():
    sql = get_schema_sql()
    required_tables = [
        "documents",
        "document_chunks",
        "normalized_entities",
        "relationships",
        "relationship_evidence",
    ]
    for table in required_tables:
        pattern = rf"CREATE TABLE IF NOT EXISTS {table}\s*\("
        assert re.search(pattern, sql, re.IGNORECASE), f"Missing CREATE TABLE IF NOT EXISTS for {table}"


def test_schema_documents_columns():
    sql = get_schema_sql()
    documents_cols = [
        "id UUID PRIMARY KEY",
        "doi VARCHAR UNIQUE",
        "pmid VARCHAR",
        "title TEXT",
        "journal VARCHAR",
        "published_date DATE",
        "parsed_json JSONB",
    ]
    for col in documents_cols:
        assert col in sql, f"Missing column definition: {col}"


def test_schema_document_chunks_columns():
    sql = get_schema_sql()
    chunks_cols = [
        "id UUID PRIMARY KEY",
        "document_id UUID",
        "REFERENCES documents(id)",
        "chunk_index INT",
        "content TEXT",
        "token_count INT",
    ]
    for col in chunks_cols:
        assert col in sql, f"Missing chunk column definition: {col}"


def test_schema_normalized_entities_columns():
    sql = get_schema_sql()
    entity_cols = [
        "id UUID PRIMARY KEY",
        "canonical_id VARCHAR UNIQUE",
        "name VARCHAR",
        "entity_type VARCHAR",
        "ontology_source VARCHAR",
        "synonyms TEXT[]",
    ]
    for col in entity_cols:
        assert col in sql, f"Missing normalized_entity column definition: {col}"


def test_schema_relationships_columns():
    sql = get_schema_sql()
    rel_cols = [
        "id UUID PRIMARY KEY",
        "source_entity_id UUID",
        "target_entity_id UUID",
        "REFERENCES normalized_entities(id)",
        "relationship_type VARCHAR",
        "confidence_score DOUBLE PRECISION",
        "curation_status VARCHAR",
        "source_type VARCHAR",
    ]
    for col in rel_cols:
        assert col in sql, f"Missing relationship column definition: {col}"


def test_schema_relationship_evidence_columns():
    sql = get_schema_sql()
    ev_cols = [
        "id UUID PRIMARY KEY",
        "relationship_id UUID",
        "chunk_id UUID",
        "REFERENCES relationships(id)",
        "REFERENCES document_chunks(id)",
        "confidence_score DOUBLE PRECISION",
        "citation_text TEXT",
    ]
    for col in ev_cols:
        assert col in sql, f"Missing relationship_evidence column definition: {col}"


def test_schema_indexes():
    sql = get_schema_sql()
    required_indexes = [
        "idx_documents_doi",
        "idx_documents_pmid",
        "idx_normalized_entities_canonical_id",
        "idx_normalized_entities_name",
        "idx_relationships_source_target",
        "idx_relationships_curation_confidence",
    ]
    for idx in required_indexes:
        assert idx in sql, f"Missing required index: {idx}"

    # Specific index target verification
    assert "idx_documents_doi ON documents(doi)" in sql
    assert "idx_documents_pmid ON documents(pmid)" in sql
    assert "idx_normalized_entities_canonical_id ON normalized_entities(canonical_id)" in sql
    assert "idx_normalized_entities_name ON normalized_entities(name)" in sql
    assert "idx_relationships_source_target ON relationships(source_entity_id, target_entity_id)" in sql
    assert "idx_relationships_curation_confidence ON relationships(curation_status, confidence_score)" in sql


def test_schema_idempotency_keywords():
    sql = get_schema_sql()
    lines = sql.splitlines()
    for line in lines:
        clean = line.strip().upper()
        if clean.startswith("CREATE TABLE"):
            assert "IF NOT EXISTS" in clean, f"Non-idempotent line: {line}"
        if clean.startswith("CREATE INDEX"):
            assert "IF NOT EXISTS" in clean, f"Non-idempotent line: {line}"


def test_init_db_schema_mock_execution():
    mock_conn = MagicMock()
    init_db_schema(mock_conn)

    sql_expected = get_schema_sql()
    mock_conn.execute.assert_called_once_with(sql_expected)
    mock_conn.commit.assert_called_once()


def test_sql_syntax_validity():
    sql = get_schema_sql()
    statements = [s.strip() for s in sql.split(";") if s.strip() and not s.strip().startswith("--")]
    
    for stmt in statements:
        # Check balanced parentheses
        open_parens = stmt.count("(")
        close_parens = stmt.count(")")
        assert open_parens == close_parens, f"Unbalanced parentheses in statement: {stmt[:50]}..."

        # Must start with CREATE TABLE or CREATE INDEX
        assert stmt.upper().startswith("CREATE TABLE") or stmt.upper().startswith("CREATE INDEX"), (
            f"Statement does not start with CREATE TABLE/INDEX: {stmt[:50]}..."
        )
