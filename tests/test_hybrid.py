# ruff: noqa: E402
"""Unit tests for the HybridRetriever class with mocked modules."""

from unittest.mock import MagicMock
import pytest

# Import global mocks to ensure clean test isolation
from tests.conftest import mock_st

from app.retrieval.embeddings import EmbeddingClient
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector import LanceDBClient


@pytest.fixture
def mock_clients():
    vector_client = MagicMock(spec=LanceDBClient)
    embedding_client = MagicMock(spec=EmbeddingClient)

    # Mock table and DB connection
    mock_table = MagicMock()
    mock_db = MagicMock()
    vector_client.connect.return_value = mock_db
    vector_client.init_table.return_value = mock_table

    # Default dense search return
    vector_client.search_vector.return_value = []

    # Default FTS search return
    mock_table.search.return_value.limit.return_value.to_list.return_value = []

    return vector_client, embedding_client, mock_table


def test_hybrid_retriever_init(mock_clients):
    vector_client, embedding_client, mock_table = mock_clients

    retriever = HybridRetriever(
        vector_client=vector_client,
        embedding_client=embedding_client,
        reranker_model_name="dummy-reranker"
    )

    assert retriever.vector_client == vector_client
    assert retriever.embedding_client == embedding_client
    assert retriever.reranker_model_name == "dummy-reranker"

    # Verify FTS index creation attempt
    mock_table.create_fts_index.assert_called_with("content", exist_ok=True)


def test_ensure_fts_index_fallback(mock_clients):
    vector_client, embedding_client, mock_table = mock_clients

    # Force TypeError on exist_ok=True
    def side_effect(col, exist_ok=None):
        if exist_ok is not None:
            raise TypeError("unexpected keyword argument 'exist_ok'")
        return MagicMock()
    mock_table.create_fts_index.side_effect = side_effect

    retriever = HybridRetriever(
        vector_client=vector_client,
        embedding_client=embedding_client
    )
    assert retriever is not None

    # Verify fallback call without exist_ok was executed
    mock_table.create_fts_index.assert_any_call("content")


def test_retrieve_dense_and_sparse(mock_clients):
    vector_client, embedding_client, mock_table = mock_clients

    # Mock dense results
    dense_results = [
        {
            "id": "chunk-1",
            "document_id": "doc-1",
            "chunk_index": 0,
            "content": "Epstein-Barr virus is associated with lymphoma.",
            "pmid": "12345",
            "doi": "10.1000/1",
            "title": "EBV Study 1",
            "score": 0.8,
        },
        {
            "id": "chunk-2",
            "document_id": "doc-2",
            "chunk_index": 1,
            "content": "EBV infects B cells.",
            "pmid": "23456",
            "doi": "10.1000/2",
            "title": "EBV Study 2",
            "score": 0.6,
        }
    ]
    vector_client.search_vector.return_value = dense_results
    embedding_client.embed_query.return_value = [0.1, 0.2, 0.3]

    # Mock FTS results
    fts_results = [
        {
            "id": "chunk-3",
            "document_id": "doc-3",
            "chunk_index": 0,
            "content": "Lymphoma cells and viral proteins.",
            "pmid": "34567",
            "doi": "10.1000/3",
            "title": "Lymphoma Paper",
            "_score": 12.5,
        },
        {
            "id": "chunk-1",
            "document_id": "doc-1",
            "chunk_index": 0,
            "content": "Epstein-Barr virus is associated with lymphoma.",
            "pmid": "12345",
            "doi": "10.1000/1",
            "title": "EBV Study 1",
            "_score": 8.2,
        }
    ]
    mock_table.search.return_value.limit.return_value.to_list.return_value = (
        fts_results
    )

    retriever = HybridRetriever(
        vector_client=vector_client,
        embedding_client=embedding_client
    )

    # Setup Mock CrossEncoder predictions
    mock_cross_encoder = MagicMock()
    mock_cross_encoder.predict.return_value = [0.95, 0.45, 0.15]
    mock_st.CrossEncoder.return_value = mock_cross_encoder

    results = retriever.retrieve("EBV lymphoma", top_k=2)

    # Verify inputs to embedding and search
    embedding_client.embed_query.assert_called_with("EBV lymphoma")
    vector_client.search_vector.assert_called_with([0.1, 0.2, 0.3], limit=8)
    mock_table.search.assert_called_with("EBV lymphoma", query_type="fts")

    # Verify CrossEncoder call with candidates content
    mock_cross_encoder.predict.assert_called()

    # Verify result contains top_k = 2 items sorted by cross encoder score
    assert len(results) == 2
    assert results[0]["id"] == "chunk-1"
    assert results[0]["score"] == 0.95
    assert results[1]["id"] == "chunk-3"
    assert results[1]["score"] == 0.45


def test_reciprocal_rank_fusion_logic(mock_clients):
    vector_client, embedding_client, _ = mock_clients
    retriever = HybridRetriever(
        vector_client=vector_client,
        embedding_client=embedding_client
    )

    dense = [
        {"id": "A", "score": 0.9},
        {"id": "B", "score": 0.8},
        {"id": "C", "score": 0.7},
    ]
    sparse = [
        {"id": "C", "score": 10.0},
        {"id": "A", "score": 5.0},
        {"id": "D", "score": 2.0},
    ]

    # RRF with k=60
    # A ranks: dense=1, sparse=2.
    # Score = 1/(60+1) + 1/(60+2) = 0.016393 + 0.016129 = 0.032522
    # B ranks: dense=2, sparse=None. Score = 1/(60+2) = 0.016129
    # C ranks: dense=3, sparse=1.
    # Score = 1/(60+3) + 1/(60+1) = 0.015873 + 0.016393 = 0.032266
    # D ranks: dense=None, sparse=3. Score = 1/(60+3) = 0.015873

    # Expected order: A, C, B, D
    fused = retriever._reciprocal_rank_fusion(dense, sparse, k=60)
    assert [x["id"] for x in fused] == ["A", "C", "B", "D"]
    assert fused[0]["score"] == pytest.approx(1.0 / 61 + 1.0 / 62)


def test_normalized_score_combination_logic(mock_clients):
    vector_client, embedding_client, _ = mock_clients
    retriever = HybridRetriever(
        vector_client=vector_client,
        embedding_client=embedding_client
    )

    dense = [
        {"id": "A", "score": 0.8},
        {"id": "B", "score": 0.4},
        {"id": "C", "score": 0.0},
    ]
    sparse = [
        {"id": "C", "score": 10.0},
        {"id": "A", "score": 5.0},
        {"id": "B", "score": 0.0},
    ]

    # Normalization:
    # dense scores: A=0.8, B=0.4, C=0.0 -> normalized: A=1.0, B=0.5, C=0.0
    # sparse scores: C=10.0, A=5.0, B=0.0 -> normalized: C=1.0, A=0.5, B=0.0
    # Combined (alpha=0.5):
    # A: 0.5 * 1.0 + 0.5 * 0.5 = 0.75
    # B: 0.5 * 0.5 + 0.5 * 0.0 = 0.25
    # C: 0.5 * 0.0 + 0.5 * 1.0 = 0.5

    # Expected order: A (0.75), C (0.5), B (0.25)
    combined = retriever._normalized_score_combination(dense, sparse, alpha=0.5)
    assert [x["id"] for x in combined] == ["A", "C", "B"]
    assert combined[0]["score"] == 0.75
    assert combined[1]["score"] == 0.5
    assert combined[2]["score"] == 0.25


def test_reranker_fallback_on_failure(mock_clients):
    vector_client, embedding_client, _ = mock_clients
    retriever = HybridRetriever(
        vector_client=vector_client,
        embedding_client=embedding_client
    )

    # Force reranker to raise exception
    retriever._reranker_failed = True

    dense = [{"id": "chunk-1", "content": "Viral DNA", "score": 0.9}]
    vector_client.search_vector.return_value = dense
    embedding_client.embed_query.return_value = [0.1]

    # Retrieve should fall back gracefully without error
    results = retriever.retrieve("viral", top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "chunk-1"
