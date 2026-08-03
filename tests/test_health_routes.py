"""Unit tests for FastAPI Health and Metrics endpoints."""

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.routes import get_pg_conn, get_neo4j_client
from app.api.health_routes import (
    get_lancedb_client,
    get_kuzu_engine,
    get_pg_conn_safe,
    get_lancedb_client_safe,
    get_neo4j_client_safe,
    get_kuzu_engine_safe,
)

client = TestClient(app)


@pytest.fixture
def mock_pg_conn():
    """Fixture providing a mock PostgreSQL connection."""
    conn = MagicMock()
    cur = MagicMock()
    cur.fetchone.return_value = {"documents": 3771, "chunks": 123311, "entities": 15603, "relationships": 370708}
    conn.cursor.return_value = cur
    return conn


@pytest.fixture
def mock_lancedb_client():
    """Fixture providing a mock LanceDB client."""
    db_mock = MagicMock()
    db_mock.list_tables.return_value = ["chunks"]
    client_mock = MagicMock()
    client_mock.connect.return_value = db_mock
    return client_mock


@pytest.fixture
def mock_neo4j_client():
    """Fixture providing a mock Neo4j client."""
    client_mock = MagicMock()
    client_mock.execute_query.return_value = [{"test": 1}]
    return client_mock


@pytest.fixture
def mock_kuzu_engine():
    """Fixture providing a mock KùzuDB engine."""
    engine_mock = MagicMock()
    engine_mock.execute_query.return_value = [{"test": 1}]
    return engine_mock


@pytest.fixture(autouse=True)
def setup_health_overrides(
    mock_pg_conn,
    mock_lancedb_client,
    mock_neo4j_client,
    mock_kuzu_engine,
):
    """Register FastAPI dependency overrides for health and metrics endpoints."""
    app.dependency_overrides[get_pg_conn] = lambda: mock_pg_conn
    app.dependency_overrides[get_lancedb_client] = lambda: mock_lancedb_client
    app.dependency_overrides[get_neo4j_client] = lambda: mock_neo4j_client
    app.dependency_overrides[get_kuzu_engine] = lambda: mock_kuzu_engine

    app.dependency_overrides[get_pg_conn_safe] = lambda: mock_pg_conn
    app.dependency_overrides[get_lancedb_client_safe] = lambda: mock_lancedb_client
    app.dependency_overrides[get_neo4j_client_safe] = lambda: mock_neo4j_client
    app.dependency_overrides[get_kuzu_engine_safe] = lambda: mock_kuzu_engine

    yield

    app.dependency_overrides.clear()


# --- Unit Tests for /api/v1/health ---

def test_health_all_healthy(
    mock_pg_conn,
    mock_lancedb_client,
    mock_neo4j_client,
    mock_kuzu_engine,
):
    """Verify /api/v1/health returns status 'healthy' when all 4 DBs are operational."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "components" in data

    comps = data["components"]
    assert comps["postgres"]["status"] == "healthy"
    assert comps["lancedb"]["status"] == "healthy"
    assert comps["neo4j"]["status"] == "healthy"
    assert comps["kuzu"]["status"] == "healthy"


def test_health_degraded_postgres(mock_pg_conn):
    """Verify /api/v1/health returns status 'degraded' when PostgreSQL fails."""
    mock_pg_conn.cursor.side_effect = Exception("PostgreSQL Connection Error")
    mock_pg_conn.execute.side_effect = Exception("PostgreSQL Connection Error")

    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "degraded"
    assert data["components"]["postgres"]["status"] == "unhealthy"
    assert "PostgreSQL" in data["components"]["postgres"]["details"]
    assert data["components"]["lancedb"]["status"] == "healthy"


def test_health_degraded_lancedb(mock_lancedb_client):
    """Verify /api/v1/health returns status 'degraded' when LanceDB fails."""
    mock_lancedb_client.connect.side_effect = Exception("LanceDB File Access Error")

    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "degraded"
    assert data["components"]["lancedb"]["status"] == "unhealthy"
    assert "LanceDB" in data["components"]["lancedb"]["details"]
    assert data["components"]["postgres"]["status"] == "healthy"


def test_health_degraded_neo4j(mock_neo4j_client):
    """Verify /api/v1/health returns status 'degraded' when Neo4j fails."""
    mock_neo4j_client.execute_query.side_effect = Exception("Neo4j Bolt Timeout")

    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "degraded"
    assert data["components"]["neo4j"]["status"] == "unhealthy"
    assert "Neo4j" in data["components"]["neo4j"]["details"]


def test_health_degraded_kuzu(mock_kuzu_engine):
    """Verify /api/v1/health returns status 'degraded' when KùzuDB fails."""
    mock_kuzu_engine.execute_query.side_effect = Exception("Kùzu Engine Lock Error")

    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "degraded"
    assert data["components"]["kuzu"]["status"] == "unhealthy"
    assert "KùzuDB" in data["components"]["kuzu"]["details"]


def test_health_all_unhealthy(
    mock_pg_conn,
    mock_lancedb_client,
    mock_neo4j_client,
    mock_kuzu_engine,
):
    """Verify /api/v1/health returns 'degraded' when all DBs fail."""
    mock_pg_conn.cursor.side_effect = Exception("PG Error")
    mock_lancedb_client.connect.side_effect = Exception("Lance Error")
    mock_neo4j_client.execute_query.side_effect = Exception("Neo4j Error")
    mock_kuzu_engine.execute_query.side_effect = Exception("Kuzu Error")

    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "degraded"
    for comp in ("postgres", "lancedb", "neo4j", "kuzu"):
        assert data["components"][comp]["status"] == "unhealthy"


# --- Unit Tests for /api/v1/metrics ---

def test_metrics_success(mock_pg_conn):
    """Verify /api/v1/metrics queries PostgreSQL and returns correct count numbers."""
    cur = MagicMock()
    cur.fetchone.return_value = {
        "documents": 3771,
        "chunks": 123311,
        "entities": 15603,
        "relationships": 370708,
    }
    mock_pg_conn.cursor.return_value = cur

    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()

    assert data["documents"] == 3771
    assert data["chunks"] == 123311
    assert data["entities"] == 15603
    assert data["relationships"] == 370708
    assert "timestamp" in data


def test_metrics_tuple_row(mock_pg_conn):
    """Verify /api/v1/metrics handles tuple row results from PostgreSQL."""
    cur = MagicMock()
    cur.fetchone.return_value = (10, 50, 100, 200)
    mock_pg_conn.cursor.return_value = cur

    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()

    assert data["documents"] == 10
    assert data["chunks"] == 50
    assert data["entities"] == 100
    assert data["relationships"] == 200


def test_metrics_database_error(mock_pg_conn):
    """Verify /api/v1/metrics returns HTTP 500 when PostgreSQL query fails."""
    cur = MagicMock()
    cur.execute.side_effect = Exception("Connection lost")
    mock_pg_conn.cursor.return_value = cur

    response = client.get("/api/v1/metrics")
    assert response.status_code == 500
    data = response.json()
    assert "Failed to query database metrics" in data["detail"]
