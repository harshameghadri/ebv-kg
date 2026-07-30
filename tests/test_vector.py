"""Unit tests for LanceDBClient vector store wrapper."""

import os
import pytest
from app.retrieval.vector import LanceDBClient


def test_client_initialization_defaults(monkeypatch):
    """Verify default connection settings are read from env or default."""
    monkeypatch.setenv("LANCEDB_URI", "data/test_env_lancedb/")
    client = LanceDBClient()
    assert client.uri == "data/test_env_lancedb/"
    assert client.table_name == "chunks"
    assert client.vector_dim == 1024


def test_client_initialization_override():
    """Verify parameters override defaults."""
    client = LanceDBClient(uri="data/override_db/", table_name="test_table", vector_dim=512)
    assert client.uri == "data/override_db/"
    assert client.table_name == "test_table"
    assert client.vector_dim == 512


def test_init_table_idempotency(tmp_path):
    """Verify init_table creates the table and can be run multiple times safely."""
    db_dir = tmp_path / "lancedb"
    client = LanceDBClient(uri=str(db_dir), table_name="chunks", vector_dim=4)

    # First call - creates the table
    table = client.init_table()
    assert table is not None
    assert client._table is not None

    # Check schema fields
    schema = table.schema
    assert "id" in schema.names
    assert "document_id" in schema.names
    assert "chunk_index" in schema.names
    assert "content" in schema.names
    assert "pmid" in schema.names
    assert "doi" in schema.names
    assert "title" in schema.names
    assert "vector" in schema.names

    # Second call - should load existing table idempotently
    table2 = client.init_table()
    assert table2.name == table.name


def test_add_chunks_and_search(tmp_path):
    """Verify chunks can be added (with flat and nested metadata) and searched."""
    db_dir = tmp_path / "lancedb"
    client = LanceDBClient(uri=str(db_dir), table_name="chunks", vector_dim=3)
    client.init_table()

    chunks = [
        {
            "id": "chunk-1",
            "document_id": "doc-1",
            "chunk_index": 0,
            "content": "Epstein-Barr virus (EBV) infects B cells.",
            "pmid": "11111",
            "doi": "10.1000/1",
            "title": "EBV B Cells",
            "vector": [1.0, 0.0, 0.0],
        },
        {
            "id": "chunk-2",
            "document_id": "doc-2",
            "chunk_index": 1,
            "content": "EBV is associated with Burkitt lymphoma.",
            "metadata": {
                "pmid": "22222",
                "doi": "10.1000/2",
                "title": "EBV Lymphoma Study",
            },
            "vector": [0.0, 1.0, 0.0],
        },
    ]

    client.add_chunks(chunks)

    # Search for vectors close to first chunk
    query_vector = [1.0, 0.1, 0.0]
    results = client.search_vector(query_vector, limit=2, metric="l2")

    assert len(results) == 2
    # The first result should be chunk-1 since it's closer to [1.0, 0.1, 0.0] than chunk-2
    assert results[0]["id"] == "chunk-1"
    assert results[0]["document_id"] == "doc-1"
    assert results[0]["chunk_index"] == 0
    assert results[0]["content"] == "Epstein-Barr virus (EBV) infects B cells."
    assert results[0]["pmid"] == "11111"
    assert results[0]["doi"] == "10.1000/1"
    assert results[0]["title"] == "EBV B Cells"
    assert results[0]["score"] > 0.9  # L2 distance should be tiny, score close to 1.0

    # The second result should be chunk-2
    assert results[1]["id"] == "chunk-2"
    assert results[1]["pmid"] == "22222"
    assert results[1]["doi"] == "10.1000/2"
    assert results[1]["title"] == "EBV Lymphoma Study"


def test_search_vector_metrics(tmp_path):
    """Verify searches using different metrics (l2, cosine, dot)."""
    db_dir = tmp_path / "lancedb"
    client = LanceDBClient(uri=str(db_dir), table_name="chunks", vector_dim=3)
    client.init_table()

    # Unit normalized vectors
    chunks = [
        {
            "id": "chunk-a",
            "document_id": "doc-a",
            "chunk_index": 0,
            "content": "Content A",
            "vector": [1.0, 0.0, 0.0],
        },
        {
            "id": "chunk-b",
            "document_id": "doc-b",
            "chunk_index": 1,
            "content": "Content B",
            "vector": [0.0, 1.0, 0.0],
        },
    ]
    client.add_chunks(chunks)

    # Cosine search
    res_cosine = client.search_vector([1.0, 0.0, 0.0], limit=1, metric="cosine")
    assert len(res_cosine) == 1
    assert res_cosine[0]["id"] == "chunk-a"
    # Cosine similarity for exact match is 1.0
    assert pytest.approx(res_cosine[0]["score"], abs=1e-5) == 1.0

    # Dot product search
    res_dot = client.search_vector([1.0, 0.0, 0.0], limit=1, metric="dot")
    assert len(res_dot) == 1
    assert res_dot[0]["id"] == "chunk-a"
    # Dot product for exact match of normalized vectors is 1.0
    assert pytest.approx(res_dot[0]["score"], abs=1e-5) == 1.0


def test_clear_table(tmp_path):
    """Verify that clear_table drops the table and resets client state."""
    db_dir = tmp_path / "lancedb"
    client = LanceDBClient(uri=str(db_dir), table_name="chunks", vector_dim=3)
    
    # Init table and add content
    client.init_table()
    client.add_chunks([
        {
            "id": "chunk-1",
            "document_id": "doc-1",
            "chunk_index": 0,
            "content": "Content",
            "vector": [1.0, 0.0, 0.0],
        }
    ])
    
    # Verify table exists in DB
    db = client.connect()
    tables = db.list_tables()
    if not isinstance(tables, list):
        tables = getattr(tables, "tables", tables)
    assert "chunks" in tables

    # Clear table
    client.clear_table()
    assert client._table is None
    
    tables_post = db.list_tables()
    if not isinstance(tables_post, list):
        tables_post = getattr(tables_post, "tables", tables_post)
    assert "chunks" not in tables_post

    # Verify we can clear again safely (idempotent)
    client.clear_table()
    assert client._table is None
