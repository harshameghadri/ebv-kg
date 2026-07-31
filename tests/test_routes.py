"""Unit tests for FastAPI REST API routes."""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
import uuid
import datetime

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
    # Mock transaction manager
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
    """Register FastAPI dependency overrides."""
    app.dependency_overrides[get_pg_conn] = lambda: mock_pg_conn
    app.dependency_overrides[get_hybrid_retriever] = lambda: mock_hybrid_retriever
    app.dependency_overrides[get_graph_retriever] = lambda: mock_graph_retriever
    app.dependency_overrides[get_claude_client] = lambda: mock_claude_client
    app.dependency_overrides[get_neo4j_client] = lambda: mock_neo4j_client
    
    yield
    
    app.dependency_overrides.clear()

# --- Unit Tests ---

def test_query_hybrid(mock_hybrid_retriever, mock_graph_retriever, mock_claude_client):
    """Verify POST /api/query/hybrid successfully calls retrievers and LLM client."""
    mock_hybrid_retriever.retrieve.return_value = [
        {
            "id": "chunk-1",
            "document_id": "doc-1",
            "chunk_index": 0,
            "content": "EBV infects B cells via CD21 receptor interaction.",
            "pmid": "123456",
            "doi": "10.1002/ebv.1",
            "title": "EBV Infection pathways",
            "score": 0.85,
        }
    ]
    mock_graph_retriever.retrieve_graph_context.return_value = "EBV -[binds]-> CD21"
    mock_claude_client.synthesize.return_value = {
        "answer": "EBV binds to CD21 [1] to infect B cells.",
        "confidence": 0.92,
        "citations": [
            {
                "source_index": 1,
                "chunk_id": "chunk-1",
                "pmid": "123456",
                "doi": "10.1002/ebv.1",
            }
        ],
    }

    response = client.post(
        "/api/query/hybrid",
        json={
            "query": "How does EBV infect B cells?",
            "top_k": 3,
            "search_type": "hybrid",
            "include_citations": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "How does EBV infect B cells?"
    assert "CD21" in data["answer"]
    assert data["confidence"] == 0.92
    assert len(data["retrieved_documents"]) == 1
    assert data["retrieved_documents"][0]["id"] == "chunk-1"
    assert len(data["citations"]) == 1
    assert data["citations"][0]["chunk_id"] == "chunk-1"
    assert data["generation_time_s"] >= 0.0

    mock_hybrid_retriever.retrieve.assert_called_once_with(
        query="How does EBV infect B cells?", top_k=3
    )
    mock_graph_retriever.retrieve_graph_context.assert_called_once_with(
        query="How does EBV infect B cells?"
    )
    mock_claude_client.synthesize.assert_called_once()

def test_explore_graph(mock_graph_retriever, mock_neo4j_client):
    """Verify GET /api/graph/explore/{entity_id} successfully queries graph neighborhood."""
    mock_neo4j_client.execute_query.return_value = [
        {"canonical_id": "HGNC:11985"}
    ]
    
    mock_graph_retriever.get_neighborhood.return_value = {
        "entities": [
            {"canonical_id": "HGNC:11985", "name": "TP53", "entity_type": "GENE"}
        ],
        "relationships": [
            {
                "id": "rel-1",
                "source_id": "HGNC:11985",
                "target_id": "HGNC:11986",
                "rel_type": "INTERACTS_WITH",
                "confidence_score": 0.85,
                "curation_status": "APPROVED",
            }
        ],
        "papers": [
            {
                "doi": "10.1002/ebv.1",
                "pmid": "123456",
                "title": "EBV and TP53",
                "journal": "Nature",
                "published_date": datetime.date(2023, 1, 1),
            }
        ],
        "mentions": [
            {
                "paper_doi": "10.1002/ebv.1",
                "entity_id": "HGNC:11985",
                "confidence_score": 0.95,
            }
        ],
    }

    response = client.get("/api/graph/explore/TP53")

    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "relationships" in data

    nodes = data["nodes"]
    assert len(nodes) == 2  # TP53 entity + 1 paper
    ent_node = next(n for n in nodes if n["label"] == "Entity")
    assert ent_node["name"] == "TP53"
    assert ent_node["id"] == "HGNC:11985"
    
    paper_node = next(n for n in nodes if n["label"] == "Paper")
    assert paper_node["id"] == "10.1002/ebv.1"
    assert paper_node["title"] == "EBV and TP53"

    relationships = data["relationships"]
    assert len(relationships) == 2
    rel_int = next(r for r in relationships if r["type"] == "INTERACTS_WITH")
    assert rel_int["source"] == "HGNC:11985"
    assert rel_int["confidence_score"] == 0.85

    rel_men = next(r for r in relationships if r["type"] == "MENTIONS")
    assert rel_men["source"] == "10.1002/ebv.1"
    assert rel_men["target"] == "HGNC:11985"

def test_curation_pending(mock_pg_conn):
    """Verify GET /api/curation/pending returns pending rows in PostgreSQL."""
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
    assert len(data) == 1
    assert data[0]["relationship_id"] == str(mock_uuid)
    assert data[0]["source_canonical_id"] == "HGNC:11985"
    assert data[0]["source_name"] == "TP53"
    assert data[0]["target_name"] == "MDM2"
    assert data[0]["relationship_type"] == "INHIBITS"
    assert data[0]["confidence_score"] == 0.95
    assert data[0]["citation_text"] == "MDM2 directly inhibits TP53 function."

def test_curation_action_approve(mock_pg_conn, mock_neo4j_client):
    """Verify POST /api/curation/action (APPROVE) executes db transaction and syncs to Neo4j."""
    mock_cursor = MagicMock()
    mock_pg_conn.cursor.return_value.__enter__.return_value = mock_cursor

    rel_uuid = str(uuid.uuid4())
    src_uuid = uuid.uuid4()
    tgt_uuid = uuid.uuid4()
    paper_uuid = uuid.uuid4()

    # Mock cursor fetch returns:
    # 1. relationship row (checking if it exists)
    # 2. source entity row
    # 3. target entity row
    mock_cursor.fetchone.side_effect = [
        {
            "source_entity_id": src_uuid,
            "target_entity_id": tgt_uuid,
            "relationship_type": "INTERACTS_WITH",
            "confidence_score": 0.82,
            "source_type": "LITERATURE",
        },
        {
            "id": src_uuid,
            "canonical_id": "HGNC:1",
            "name": "E1",
            "entity_type": "GENE",
            "ontology_source": "HGNC",
            "synonyms": ["Syn1"],
        },
        {
            "id": tgt_uuid,
            "canonical_id": "HGNC:2",
            "name": "E2",
            "entity_type": "GENE",
            "ontology_source": "HGNC",
            "synonyms": ["Syn2"],
        },
    ]

    # Mock cursor fetchall returns:
    # 1. mentions query response
    # 2. papers query response
    mock_cursor.fetchall.side_effect = [
        [
            {
                "source_doi": "10.1002/ebv.1",
                "target_canonical_id": "HGNC:1",
                "confidence_score": 0.82,
            }
        ],
        [
            {
                "id": paper_uuid,
                "doi": "10.1002/ebv.1",
                "pmid": "12345",
                "title": "EBV Study",
                "journal": "J. Virol.",
                "published_date": datetime.date(2023, 5, 10),
            }
        ],
    ]

    response = client.post(
        "/api/curation/action",
        json={"relationship_id": rel_uuid, "action": "APPROVE"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "success", "curation_status": "APPROVED"}

    # Check Entity Nodes sync
    mock_neo4j_client.bulk_upsert_nodes.assert_any_call(
        label="Entity",
        nodes=[
            {
                "id": str(src_uuid),
                "canonical_id": "HGNC:1",
                "name": "E1",
                "entity_type": "GENE",
                "ontology_source": "HGNC",
                "synonyms": ["Syn1"],
            },
            {
                "id": str(tgt_uuid),
                "canonical_id": "HGNC:2",
                "name": "E2",
                "entity_type": "GENE",
                "ontology_source": "HGNC",
                "synonyms": ["Syn2"],
            },
        ],
        id_property="canonical_id",
    )

    # Check Relationship Edge sync
    mock_neo4j_client.bulk_upsert_edges.assert_any_call(
        rel_type="INTERACTS_WITH",
        edges=[
            {
                "id": rel_uuid,
                "source_canonical_id": "HGNC:1",
                "target_canonical_id": "HGNC:2",
                "confidence_score": 0.82,
                "curation_status": "APPROVED",
                "source_type": "LITERATURE",
            }
        ],
        source_label="Entity",
        target_label="Entity",
        source_key="canonical_id",
        target_key="canonical_id",
    )

    # Check Paper Nodes sync
    mock_neo4j_client.bulk_upsert_nodes.assert_any_call(
        label="Paper",
        nodes=[
            {
                "id": str(paper_uuid),
                "doi": "10.1002/ebv.1",
                "pmid": "12345",
                "title": "EBV Study",
                "journal": "J. Virol.",
                "published_date": "2023-05-10",
            }
        ],
        id_property="doi",
    )

    # Check Mentions Edge sync
    mock_neo4j_client.bulk_upsert_edges.assert_any_call(
        rel_type="MENTIONS",
        edges=[
            {
                "source_doi": "10.1002/ebv.1",
                "target_canonical_id": "HGNC:1",
                "confidence_score": 0.82,
            }
        ],
        source_label="Paper",
        target_label="Entity",
        source_key="doi",
        target_key="canonical_id",
    )

def test_curation_action_reject(mock_pg_conn, mock_neo4j_client):
    """Verify POST /api/curation/action (REJECT) rejects in db and does NOT sync to Neo4j."""
    mock_cursor = MagicMock()
    mock_pg_conn.cursor.return_value.__enter__.return_value = mock_cursor

    rel_uuid = str(uuid.uuid4())
    src_uuid = uuid.uuid4()
    tgt_uuid = uuid.uuid4()

    mock_cursor.fetchone.return_value = {
        "source_entity_id": src_uuid,
        "target_entity_id": tgt_uuid,
        "relationship_type": "INTERACTS_WITH",
        "confidence_score": 0.82,
        "source_type": "LITERATURE",
    }

    response = client.post(
        "/api/curation/action",
        json={"relationship_id": rel_uuid, "action": "REJECT"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "success", "curation_status": "REJECTED"}

    # Neo4j operations should NOT be triggered
    mock_neo4j_client.bulk_upsert_nodes.assert_not_called()
    mock_neo4j_client.bulk_upsert_edges.assert_not_called()

def test_curation_status_agg(mock_pg_conn):
    """Verify GET /api/admin/curation-status fetches and aggregates PostgreSQL statuses."""
    mock_cursor = MagicMock()
    mock_pg_conn.cursor.return_value.__enter__.return_value = mock_cursor

    mock_cursor.fetchall.return_value = [
        {"curation_status": "APPROVED", "count": 25},
        {"curation_status": "PENDING", "count": 14},
        {"curation_status": "REJECTED", "count": 6},
    ]

    response = client.get("/api/admin/curation-status")

    assert response.status_code == 200
    data = response.json()
    assert data["approved"] == 25
    assert data["pending"] == 14
    assert data["rejected"] == 6
