import json
import os
from unittest.mock import MagicMock, patch
import pytest
from app.evaluation.benchmark import RAGEvaluator, DEFAULT_GOLDEN_QUERIES


def test_evaluator_initialization_defaults():
    """Verify evaluator initializes with default golden queries when no args provided."""
    evaluator = RAGEvaluator()
    assert evaluator.queries == DEFAULT_GOLDEN_QUERIES
    assert len(evaluator.queries) >= 10


def test_evaluator_initialization_custom_queries():
    """Verify evaluator initializes with custom queries list."""
    custom_queries = [
        {
            "query": "Custom query test",
            "pmids": ["99999"],
            "dois": ["10.1000/custom"],
            "content": "Custom content details.",
        }
    ]
    evaluator = RAGEvaluator(queries=custom_queries)
    assert evaluator.queries == custom_queries


def test_evaluator_initialization_from_file(tmp_path):
    """Verify evaluator loads queries from a JSON file path."""
    custom_queries = [
        {
            "query": "File loaded query",
            "pmids": ["88888"],
            "dois": ["10.1000/file"],
            "content": "Loaded from file.",
        }
    ]
    query_file = tmp_path / "queries.json"
    with open(query_file, "w", encoding="utf-8") as f:
        json.dump(custom_queries, f)

    evaluator = RAGEvaluator(query_file_path=str(query_file))
    assert evaluator.queries == custom_queries


def test_evaluate_embeddings_perfect_match():
    """Verify metric calculations when the target document is retrieved at rank 1."""
    custom_queries = [
        {
            "query": "What is the function of EBNA1?",
            "pmids": ["12300001"],
            "dois": ["10.1128/jvi.1"],
        }
    ]
    evaluator = RAGEvaluator(queries=custom_queries)

    mock_embedding_client = MagicMock()
    mock_embedding_client.embed_query.return_value = [0.1, 0.2, 0.3]

    mock_vector_client = MagicMock()
    # Mock return of search_vector (top 5 retrieved chunks)
    mock_vector_client.search_vector.return_value = [
        {
            "id": "bench-chunk-0",
            "pmid": "12300001",
            "doi": "10.1128/jvi.1",
            "score": 0.95,
        },
        {
            "id": "distractor-0",
            "pmid": "distractor-pmid-0",
            "doi": "distractor-doi-0",
            "score": 0.80,
        },
        {
            "id": "distractor-1",
            "pmid": "distractor-pmid-1",
            "doi": "distractor-doi-1",
            "score": 0.75,
        },
        {
            "id": "distractor-2",
            "pmid": "distractor-pmid-2",
            "doi": "distractor-doi-2",
            "score": 0.70,
        },
        {
            "id": "distractor-3",
            "pmid": "distractor-pmid-3",
            "doi": "distractor-doi-3",
            "score": 0.65,
        },
    ]

    results = evaluator.evaluate_embeddings(
        mock_vector_client, mock_embedding_client, k=5
    )

    # 1 relevant retrieved in top 5 -> precision = 1 / 5 = 0.2
    assert pytest.approx(results["mean_precision"]) == 0.2
    # 1 expected relevant, 1 retrieved -> recall = 1 / 1 = 1.0
    assert pytest.approx(results["mean_recall"]) == 1.0
    # First relevant chunk is at rank 1 -> Reciprocal Rank = 1 / 1 = 1.0
    assert pytest.approx(results["mrr"]) == 1.0

    mock_embedding_client.embed_query.assert_called_once_with(
        "What is the function of EBNA1?"
    )
    mock_vector_client.search_vector.assert_called_once_with(
        [0.1, 0.2, 0.3], limit=5
    )


def test_evaluate_embeddings_second_match():
    """Verify metric calculations when the target document is retrieved at rank 2."""
    custom_queries = [
        {
            "query": "Role of LMP1 in cancer",
            "pmids": ["12300002"],
            "dois": [],
        }
    ]
    evaluator = RAGEvaluator(queries=custom_queries)

    mock_embedding_client = MagicMock()
    mock_embedding_client.embed_query.return_value = [0.4, 0.5, 0.6]

    mock_vector_client = MagicMock()
    mock_vector_client.search_vector.return_value = [
        {
            "id": "distractor-0",
            "pmid": "distractor-pmid-0",
            "doi": "distractor-doi-0",
            "score": 0.90,
        },
        {
            "id": "bench-chunk-1",
            "pmid": "12300002",
            "doi": "10.1128/jvi.2",
            "score": 0.85,
        },
        {
            "id": "distractor-1",
            "pmid": "distractor-pmid-1",
            "doi": "distractor-doi-1",
            "score": 0.80,
        },
        {
            "id": "distractor-2",
            "pmid": "distractor-pmid-2",
            "doi": "distractor-doi-2",
            "score": 0.70,
        },
        {
            "id": "distractor-3",
            "pmid": "distractor-pmid-3",
            "doi": "distractor-doi-3",
            "score": 0.60,
        },
    ]

    results = evaluator.evaluate_embeddings(
        mock_vector_client, mock_embedding_client, k=5
    )

    # 1 relevant retrieved in top 5 -> precision = 1 / 5 = 0.2
    assert pytest.approx(results["mean_precision"]) == 0.2
    # 1 expected relevant, 1 retrieved -> recall = 1 / 1 = 1.0
    assert pytest.approx(results["mean_recall"]) == 1.0
    # First relevant chunk is at rank 2 -> Reciprocal Rank = 1 / 2 = 0.5
    assert pytest.approx(results["mrr"]) == 0.5


def test_evaluate_embeddings_no_match():
    """Verify metric calculations when no relevant documents are retrieved."""
    custom_queries = [
        {
            "query": "Multiple sclerosis and EBV",
            "pmids": ["12300003"],
            "dois": ["10.1128/jvi.3"],
        }
    ]
    evaluator = RAGEvaluator(queries=custom_queries)

    mock_embedding_client = MagicMock()
    mock_embedding_client.embed_query.return_value = [0.7, 0.8, 0.9]

    mock_vector_client = MagicMock()
    # All retrieved chunks are distractors
    mock_vector_client.search_vector.return_value = [
        {
            "id": "distractor-0",
            "pmid": "distractor-pmid-0",
            "doi": "distractor-doi-0",
            "score": 0.90,
        },
        {
            "id": "distractor-1",
            "pmid": "distractor-pmid-1",
            "doi": "distractor-doi-1",
            "score": 0.80,
        },
        {
            "id": "distractor-2",
            "pmid": "distractor-pmid-2",
            "doi": "distractor-doi-2",
            "score": 0.70,
        },
    ]

    results = evaluator.evaluate_embeddings(
        mock_vector_client, mock_embedding_client, k=3
    )

    # 0 relevant retrieved in top 3 -> precision = 0.0
    assert pytest.approx(results["mean_precision"]) == 0.0
    # 0 relevant retrieved in top 3 -> recall = 0.0
    assert pytest.approx(results["mean_recall"]) == 0.0
    # No relevant chunks -> Reciprocal Rank = 0.0
    assert pytest.approx(results["mrr"]) == 0.0


def test_populate_benchmark_data():
    """Verify benchmark data population process generates, embeds, and adds chunks."""
    custom_queries = [
        {
            "query": "EBV Entry mechanisms",
            "pmids": ["12300005"],
            "dois": ["10.1128/jvi.5"],
            "content": "Specific entry details here.",
        }
    ]
    evaluator = RAGEvaluator(queries=custom_queries)

    mock_embedding_client = MagicMock()
    # 1 custom query chunk + 10 distractors = 11 chunks total
    mock_embedding_client.embed_documents.return_value = [[0.1] * 1024] * 11

    mock_vector_client = MagicMock()

    evaluator.populate_benchmark_data(mock_vector_client, mock_embedding_client)

    mock_vector_client.clear_table.assert_called_once()
    mock_vector_client.init_table.assert_called_once()
    mock_embedding_client.embed_documents.assert_called_once()

    # Verify add_chunks is called with list of chunks
    mock_vector_client.add_chunks.assert_called_once()
    added_chunks = mock_vector_client.add_chunks.call_args[0][0]
    assert len(added_chunks) == 11
    # Check that custom query chunk is included
    assert added_chunks[0]["pmid"] == "12300005"
    assert added_chunks[0]["doi"] == "10.1128/jvi.5"
    assert added_chunks[0]["content"] == "Specific entry details here."
    assert added_chunks[0]["vector"] == [0.1] * 1024


@patch("app.retrieval.embeddings.EmbeddingClient")
@patch("app.retrieval.vector.LanceDBClient")
def test_run_comparison(mock_lancedb_client_class, mock_embedding_client_class):
    """Verify run_comparison initializes clients and returns compiled metrics dictionary."""
    # Setup mocks
    mock_embedding_client = MagicMock()
    mock_embedding_client.embed_query.return_value = [0.1, 0.2]
    mock_embedding_client_class.return_value = mock_embedding_client

    mock_vector_client = MagicMock()
    mock_lancedb_client_class.return_value = mock_vector_client

    custom_queries = [
        {
            "query": "Test query",
            "pmids": ["11111"],
            "dois": ["10.1000/test"],
            "content": "Content for testing",
        }
    ]
    evaluator = RAGEvaluator(queries=custom_queries)

    # Patch populate and evaluate methods on the evaluator
    evaluator.populate_benchmark_data = MagicMock()
    evaluator.evaluate_embeddings = MagicMock(
        return_value={"mean_precision": 0.4, "mean_recall": 0.8, "mrr": 0.5}
    )

    models = ["model-A", "model-B"]
    results = evaluator.run_comparison(model_names=models, k=5)

    assert len(results) == 2
    assert "model-A" in results
    assert "model-B" in results
    assert results["model-A"]["mean_precision"] == 0.4
    assert results["model-B"]["mrr"] == 0.5

    # Check cleanup is called
    assert mock_vector_client.clear_table.call_count == 2
