"""Unit tests for the chunk embedding pipeline (EmbeddingsPipeline)."""

from unittest.mock import MagicMock, patch
import pytest
import pyarrow as pa

from psycopg.rows import dict_row
from app.ingestion.embeddings_pipeline import EmbeddingsPipeline
from app.retrieval.embeddings import EmbeddingClient
from app.retrieval.vector import LanceDBClient


def test_pipeline_initialization():
    """Verify default clients are instantiated if none are provided."""
    with patch("app.ingestion.embeddings_pipeline.EmbeddingClient") as mock_emb_class, \
         patch("app.ingestion.embeddings_pipeline.LanceDBClient") as mock_vec_class:
        
        mock_emb = MagicMock()
        mock_vec = MagicMock()
        mock_emb_class.return_value = mock_emb
        mock_vec_class.return_value = mock_vec

        pipeline = EmbeddingsPipeline()
        assert pipeline.embedding_client == mock_emb
        assert pipeline.vector_client == mock_vec

        # Test custom clients
        custom_emb = MagicMock(spec=EmbeddingClient)
        custom_vec = MagicMock(spec=LanceDBClient)
        pipeline_custom = EmbeddingsPipeline(embedding_client=custom_emb, vector_client=custom_vec)
        assert pipeline_custom.embedding_client == custom_emb
        assert pipeline_custom.vector_client == custom_vec


def test_index_pending_chunks_all_new():
    """Verify indexing works when all chunks from PostgreSQL are new (not in LanceDB)."""
    # Mock PostgreSQL Connection & Cursor
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    
    # Mock data from PostgreSQL
    pg_rows = [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "chunk_index": 0,
            "content": "Content of chunk 1",
            "pmid": "pmid-1",
            "doi": "doi-1",
            "title": "Title 1",
        },
        {
            "chunk_id": "chunk-2",
            "document_id": "doc-1",
            "chunk_index": 1,
            "content": "Content of chunk 2",
            "pmid": "pmid-1",
            "doi": "doi-1",
            "title": "Title 1",
        }
    ]
    mock_cur.fetchall.return_value = pg_rows

    # Mock LanceDB client
    mock_vec = MagicMock(spec=LanceDBClient)
    mock_vec.table_name = "chunks"
    mock_db = MagicMock()
    mock_vec.connect.return_value = mock_db
    # Table does not exist yet (or is empty)
    mock_db.list_tables.return_value = []

    # Mock EmbeddingClient
    mock_emb = MagicMock(spec=EmbeddingClient)
    mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_emb.embed_documents.return_value = mock_embeddings

    pipeline = EmbeddingsPipeline(embedding_client=mock_emb, vector_client=mock_vec)
    total_indexed = pipeline.index_pending_chunks(mock_conn, batch_size=10)

    assert total_indexed == 2
    mock_vec.init_table.assert_called_once()
    mock_emb.embed_documents.assert_called_once_with(["Content of chunk 1", "Content of chunk 2"])
    
    # Verify exact contents added to LanceDB
    mock_vec.add_chunks.assert_called_once()
    added_chunks = mock_vec.add_chunks.call_args[0][0]
    assert len(added_chunks) == 2
    assert added_chunks[0]["id"] == "chunk-1"
    assert added_chunks[0]["vector"] == [0.1, 0.2, 0.3]
    assert added_chunks[1]["id"] == "chunk-2"
    assert added_chunks[1]["vector"] == [0.4, 0.5, 0.6]


def test_index_pending_chunks_some_existing():
    """Verify that chunks already in LanceDB are skipped."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    
    # PG returns chunk-1 and chunk-2
    pg_rows = [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "chunk_index": 0,
            "content": "Content of chunk 1",
            "pmid": "pmid-1",
            "doi": "doi-1",
            "title": "Title 1",
        },
        {
            "chunk_id": "chunk-2",
            "document_id": "doc-2",
            "chunk_index": 0,
            "content": "Content of chunk 2",
            "pmid": "pmid-2",
            "doi": "doi-2",
            "title": "Title 2",
        }
    ]
    mock_cur.fetchall.return_value = pg_rows

    # Mock LanceDBClient: 'chunk-1' already exists
    mock_vec = MagicMock(spec=LanceDBClient)
    mock_vec.table_name = "chunks"
    mock_db = MagicMock()
    mock_vec.connect.return_value = mock_db
    mock_db.list_tables.return_value = ["chunks"]
    
    mock_table = MagicMock()
    # Mock Arrow table with existing id 'chunk-1'
    mock_arrow_table = pa.table({"id": ["chunk-1"]})
    mock_table.to_arrow.return_value = mock_arrow_table
    mock_db.open_table.return_value = mock_table

    # Mock EmbeddingClient
    mock_emb = MagicMock(spec=EmbeddingClient)
    mock_emb.embed_documents.return_value = [[0.7, 0.8, 0.9]]

    pipeline = EmbeddingsPipeline(embedding_client=mock_emb, vector_client=mock_vec)
    total_indexed = pipeline.index_pending_chunks(mock_conn, batch_size=10)

    # Only chunk-2 should be indexed
    assert total_indexed == 1
    mock_emb.embed_documents.assert_called_once_with(["Content of chunk 2"])
    
    mock_vec.add_chunks.assert_called_once()
    added_chunks = mock_vec.add_chunks.call_args[0][0]
    assert len(added_chunks) == 1
    assert added_chunks[0]["id"] == "chunk-2"
    assert added_chunks[0]["vector"] == [0.7, 0.8, 0.9]


def test_index_pending_chunks_all_existing():
    """Verify that if all chunks already exist in LanceDB, nothing is processed."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    
    pg_rows = [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "chunk_index": 0,
            "content": "Content of chunk 1",
            "pmid": "pmid-1",
            "doi": "doi-1",
            "title": "Title 1",
        }
    ]
    mock_cur.fetchall.return_value = pg_rows

    mock_vec = MagicMock(spec=LanceDBClient)
    mock_vec.table_name = "chunks"
    mock_db = MagicMock()
    mock_vec.connect.return_value = mock_db
    mock_db.list_tables.return_value = ["chunks"]
    
    mock_table = MagicMock()
    mock_arrow_table = pa.table({"id": ["chunk-1"]})
    mock_table.to_arrow.return_value = mock_arrow_table
    mock_db.open_table.return_value = mock_table

    mock_emb = MagicMock(spec=EmbeddingClient)

    pipeline = EmbeddingsPipeline(embedding_client=mock_emb, vector_client=mock_vec)
    total_indexed = pipeline.index_pending_chunks(mock_conn, batch_size=10)

    assert total_indexed == 0
    mock_emb.embed_documents.assert_not_called()
    mock_vec.add_chunks.assert_not_called()


def test_index_pending_chunks_batching():
    """Verify chunks are embedded and indexed in correct batch sizes."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    
    pg_rows = [
        {"chunk_id": f"chunk-{i}", "document_id": "doc-1", "chunk_index": i, "content": f"Content {i}", "pmid": "pmid-1", "doi": "doi-1", "title": "Title 1"}
        for i in range(5)
    ]
    mock_cur.fetchall.return_value = pg_rows

    mock_vec = MagicMock(spec=LanceDBClient)
    mock_vec.table_name = "chunks"
    mock_db = MagicMock()
    mock_vec.connect.return_value = mock_db
    mock_db.list_tables.return_value = []

    mock_emb = MagicMock(spec=EmbeddingClient)
    mock_emb.embed_documents.side_effect = [
        [[0.1], [0.2]],  # batch 1
        [[0.3], [0.4]],  # batch 2
        [[0.5]]          # batch 3
    ]

    pipeline = EmbeddingsPipeline(embedding_client=mock_emb, vector_client=mock_vec)
    # Batch size 2
    total_indexed = pipeline.index_pending_chunks(mock_conn, batch_size=2)

    assert total_indexed == 5
    assert mock_emb.embed_documents.call_count == 3
    assert mock_vec.add_chunks.call_count == 3


def test_index_pending_chunks_handles_null_content():
    """Verify that chunks with content=None are replaced with empty string."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    
    pg_rows = [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "chunk_index": 0,
            "content": None,
            "pmid": "pmid-1",
            "doi": "doi-1",
            "title": "Title 1",
        }
    ]
    mock_cur.fetchall.return_value = pg_rows

    mock_vec = MagicMock(spec=LanceDBClient)
    mock_vec.table_name = "chunks"
    mock_db = MagicMock()
    mock_vec.connect.return_value = mock_db
    mock_db.list_tables.return_value = []

    mock_emb = MagicMock(spec=EmbeddingClient)
    mock_emb.embed_documents.return_value = [[0.0]]

    pipeline = EmbeddingsPipeline(embedding_client=mock_emb, vector_client=mock_vec)
    total_indexed = pipeline.index_pending_chunks(mock_conn, batch_size=10)

    assert total_indexed == 1
    mock_emb.embed_documents.assert_called_once_with([""])
    added_chunks = mock_vec.add_chunks.call_args[0][0]
    assert added_chunks[0]["content"] == ""
