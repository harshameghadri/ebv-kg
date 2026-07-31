"""Integration tests for FastAPI server and Curation Dashboard UI."""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
import uuid

from app.main import app
from app.api.routes import (
    get_pg_conn,
    get_hybrid_retriever,
    get_graph_retriever,
    get_claude_client,
    get_neo4j_client,
)

# Initialize TestClient
client = TestClient(app)

@pytest.fixture
def mock_pg_conn():
    """Fixture providing a mock PostgreSQL connection."""
    conn = MagicMock()
    conn.transaction.return_value.__enter__.return_value = conn
    return conn

@pytest.fixture
def mock_hybrid_retriever():
    """Fixture providing a mock HybridRetriever."""
    return MagicMock()

@pytest.fixture
def mock_graph_retriever():
    """Fixture providing a mock GraphRetriever."""
    return MagicMock()

@pytest.fixture
def mock_claude_client():
    """Fixture providing a mock ClaudeSynthesisClient."""
    return MagicMock()

@pytest.fixture
def mock_neo4j_client():
    """Fixture providing a mock Neo4jClient."""
    return MagicMock()

@pytest.fixture(autouse=True)
def setup_overrides(
    mock_pg_conn,
    mock_hybrid_retriever,
    mock_graph_retriever,
    mock_claude_client,
    mock_neo4j_client,
):
    """Register FastAPI dependency overrides for UI tests."""
    app.dependency_overrides[get_pg_conn] = lambda: mock_pg_conn
    app.dependency_overrides[get_hybrid_retriever] = lambda: mock_hybrid_retriever
    app.dependency_overrides[get_graph_retriever] = lambda: mock_graph_retriever
    app.dependency_overrides[get_claude_client] = lambda: mock_claude_client
    app.dependency_overrides[get_neo4j_client] = lambda: mock_neo4j_client
    
    yield
    
    app.dependency_overrides.clear()


def test_dashboard_endpoint():
    """Verify that the dashboard endpoint serves the index.html page."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "<title>EBV Knowledge System | Dashboard</title>" in response.text


def test_pending_curation_endpoint(mock_pg_conn):
    """Verify pending curation endpoint returns mocked SQL records."""
    mock_cursor = MagicMock()
    mock_pg_conn.cursor.return_value.__enter__.return_value = mock_cursor

    mock_uuid = uuid.uuid4()
    mock_cursor.fetchall.return_value = [
        {
            "relationship_id": mock_uuid,
            "source_canonical_id": "HGNC:11985",
            "source_name": "TP53",
            "target_canonical_id": "HGNC:11986",
            "target_name": "MDM2",
            "relationship_type": "INHIBITS",
            "confidence_score": 0.95,
            "citation_text": "MDM2 directly inhibits TP53 function.",
        }
    ]

    response = client.get("/api/curation/pending")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["relationship_id"] == str(mock_uuid)
    assert data[0]["source_name"] == "TP53"


def test_curation_action_endpoint(mock_pg_conn):
    """Verify curation action endpoint executes and updates DB status."""
    mock_cursor = MagicMock()
    mock_pg_conn.cursor.return_value.__enter__.return_value = mock_cursor

    # Mock relationship lookup showing it exists
    mock_cursor.fetchone.return_value = {
        "source_entity_id": uuid.uuid4(),
        "target_entity_id": uuid.uuid4(),
        "relationship_type": "INTERACTS_WITH",
        "confidence_score": 0.82,
        "source_type": "LITERATURE",
    }

    rel_id = str(uuid.uuid4())
    response = client.post("/api/curation/action", json={"relationship_id": rel_id, "action": "REJECT"})
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["curation_status"] == "REJECTED"


def test_admin_curation_status_endpoint(mock_pg_conn):
    """Verify curation status analytics endpoint queries stats."""
    mock_cursor = MagicMock()
    mock_pg_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchall.return_value = [
        {"curation_status": "APPROVED", "count": 10},
        {"curation_status": "PENDING", "count": 5},
    ]

    response = client.get("/api/admin/curation-status")
    assert response.status_code == 200
    data = response.json()
    assert data["approved"] == 10
    assert data["pending"] == 5


def test_query_hybrid_endpoint(mock_hybrid_retriever, mock_graph_retriever, mock_claude_client):
    """Verify hybrid query endpoint coordinates retrievers and synthesis."""
    mock_hybrid_retriever.retrieve.return_value = [
        {
            "id": "chunk-1",
            "document_id": "doc-1",
            "chunk_index": 0,
            "content": "EBV LMP1 mimics CD40.",
            "pmid": "123",
            "doi": "10.100",
            "title": "EBV Study",
            "score": 0.9,
        }
    ]
    mock_graph_retriever.retrieve_graph_context.return_value = "LMP1 -[mimics]-> CD40"
    mock_claude_client.synthesize.return_value = {
        "answer": "LMP1 mimics CD40.",
        "confidence": 0.95,
        "citations": [],
    }

    response = client.post("/api/query/hybrid", json={"query": "What does LMP1 mimic?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "LMP1 mimics CD40."
    assert data["confidence"] == 0.95
    assert len(data["retrieved_documents"]) == 1
