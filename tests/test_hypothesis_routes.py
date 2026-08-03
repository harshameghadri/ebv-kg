"""Unit tests for FastAPI Hypothesis REST API routes."""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
import uuid

from app.main import app
from app.api.routes import (
    get_pg_conn,
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
def mock_neo4j_client():
    """Fixture providing a mock Neo4jClient."""
    return MagicMock()


@pytest.fixture(autouse=True)
def setup_overrides(mock_pg_conn, mock_neo4j_client):
    """Register FastAPI dependency overrides."""
    app.dependency_overrides[get_pg_conn] = lambda: mock_pg_conn
    app.dependency_overrides[get_neo4j_client] = lambda: mock_neo4j_client

    yield

    app.dependency_overrides.clear()


def test_niche_overlap_neo4j_get_success(mock_neo4j_client):
    """Verify GET /api/v1/hypothesis/niche-overlap queries Neo4j and returns structured JSON."""
    mock_neo4j_client.execute_query.return_value = [
        {
            "cell_state_id": "CL:0000959",
            "cell_state_name": "Atypical B Cell",
            "raw_diseases": [
                {"id": "MONDO:0005301", "name": "Multiple Sclerosis", "confidence_score": 0.95},
                {"id": "MONDO:0005009", "name": "Burkitt Lymphoma", "confidence_score": 0.85},
            ],
            "raw_genes": [
                {"id": "HGNC:1633", "name": "CD19", "symbol": "CD19"},
                {"id": "HGNC:6142", "name": "ITGAX", "symbol": "ITGAX"},
                {"id": "HGNC:11599", "name": "TBX21", "symbol": "TBX21"},
            ],
            "overlap_confidence": 0.90,
            "silo_count": 2,
        }
    ]

    response = client.get("/api/v1/hypothesis/niche-overlap")

    assert response.status_code == 200
    data = response.json()
    assert data["total_results"] == 1
    assert len(data["overlaps"]) == 1

    overlap = data["overlaps"][0]
    assert overlap["cell_state_id"] == "CL:0000959"
    assert overlap["cell_state_name"] == "Atypical B Cell"
    assert overlap["silo_count"] == 2
    assert overlap["overlap_confidence"] == 0.90

    # Verify connected diseases across silos
    diseases = overlap["connected_diseases"]
    assert len(diseases) == 2
    disease_names = [d["name"] for d in diseases]
    assert "Multiple Sclerosis" in disease_names
    assert "Burkitt Lymphoma" in disease_names

    # Verify marker genes
    genes = overlap["marker_genes"]
    assert len(genes) == 3
    gene_symbols = [g["symbol"] for g in genes]
    assert "CD19" in gene_symbols
    assert "ITGAX" in gene_symbols
    assert "TBX21" in gene_symbols

    mock_neo4j_client.execute_query.assert_called_once()


def test_niche_overlap_post_success(mock_neo4j_client):
    """Verify POST /api/v1/hypothesis/niche-overlap accepts JSON payload filters."""
    mock_neo4j_client.execute_query.return_value = [
        {
            "cell_state_id": "CL:0000959",
            "cell_state_name": "Atypical B Cell",
            "raw_diseases": [
                {"id": "MONDO:0005301", "name": "Multiple Sclerosis", "confidence_score": 0.92},
                {"id": "MONDO:0005009", "name": "Burkitt Lymphoma", "confidence_score": 0.88},
            ],
            "raw_genes": [
                {"id": "HGNC:1633", "name": "CD19", "symbol": "CD19"},
            ],
            "overlap_confidence": 0.90,
            "silo_count": 2,
        }
    ]

    payload = {
        "diseases": ["Multiple Sclerosis", "Burkitt Lymphoma"],
        "min_confidence": 0.8,
        "limit": 10,
        "source": "neo4j",
    }

    response = client.post("/api/v1/hypothesis/niche-overlap", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["total_results"] == 1
    assert data["overlaps"][0]["cell_state_name"] == "Atypical B Cell"

    # Ensure Neo4j query parameters included disease filter
    call_args = mock_neo4j_client.execute_query.call_args
    query_str, params = call_args[0]
    assert "diseases" in params
    assert params["min_confidence"] == 0.8
    assert params["limit"] == 10


def test_niche_overlap_postgres_fallback(mock_neo4j_client, mock_pg_conn):
    """Verify fallback to PostgreSQL when Neo4j returns empty results in auto mode."""
    mock_neo4j_client.execute_query.return_value = []

    mock_cursor = MagicMock()
    mock_pg_conn.cursor.return_value.__enter__.return_value = mock_cursor

    cs_uuid = uuid.uuid4()
    # Mock PostgreSQL cursor calls:
    # 1. Fetch cell states with >= 2 diseases
    # 2. Fetch diseases for cell state
    # 3. Fetch marker genes for cell state
    mock_cursor.fetchall.side_effect = [
        [
            {
                "internal_cs_id": cs_uuid,
                "cell_state_id": "CL:0000959",
                "cell_state_name": "Atypical B Cell",
            }
        ],
        [
            {"id": "MONDO:0005301", "name": "Multiple Sclerosis", "confidence_score": 0.95},
            {"id": "MONDO:0005009", "name": "Burkitt Lymphoma", "confidence_score": 0.85},
        ],
        [
            {"id": "HGNC:1633", "name": "CD19"},
            {"id": "HGNC:6142", "name": "ITGAX"},
        ],
    ]

    response = client.get("/api/v1/hypothesis/niche-overlap?source=postgres")

    assert response.status_code == 200
    data = response.json()
    assert data["total_results"] == 1

    overlap = data["overlaps"][0]
    assert overlap["cell_state_name"] == "Atypical B Cell"
    assert overlap["silo_count"] == 2
    assert overlap["overlap_confidence"] == 0.90
    assert len(overlap["connected_diseases"]) == 2
    assert len(overlap["marker_genes"]) == 2


def test_niche_overlap_empty_results(mock_neo4j_client, mock_pg_conn):
    """Verify endpoint returns total_results: 0 when no niche overlaps exist."""
    mock_neo4j_client.execute_query.return_value = []
    mock_cursor = MagicMock()
    mock_pg_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchall.return_value = []

    response = client.get("/api/v1/hypothesis/niche-overlap")

    assert response.status_code == 200
    data = response.json()
    assert data["total_results"] == 0
    assert data["overlaps"] == []


def test_niche_overlap_explicit_neo4j_error(mock_neo4j_client):
    """Verify explicit source=neo4j returns 500 error on Neo4j failure."""
    mock_neo4j_client.execute_query.side_effect = Exception("Neo4j cluster unavailable")

    response = client.get("/api/v1/hypothesis/niche-overlap?source=neo4j")

    assert response.status_code == 500
    assert "Neo4j query error" in response.json()["detail"]
