"""FastAPI Health and Metrics Router for the EBV Knowledge System."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.retrieval.vector import LanceDBClient
from app.materialization.neo4j_client import Neo4jClient
from app.materialization.kuzu_engine import KuzuEngine
from app.api.routes import get_pg_conn, get_neo4j_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Health & Metrics"])


# --- Pydantic Schemas ---

class ComponentHealth(BaseModel):
    status: str = Field(..., description="Component status: 'healthy' or 'unhealthy'")
    details: Optional[str] = Field(default=None, description="Status details or error description")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall health status: 'healthy' or 'degraded'")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")
    components: Dict[str, ComponentHealth] = Field(..., description="Individual component statuses")


class MetricsResponse(BaseModel):
    documents: int = Field(..., description="Total document count")
    chunks: int = Field(..., description="Total document chunk count")
    entities: int = Field(..., description="Total normalized entity count")
    relationships: int = Field(..., description="Total relationship count")
    total_documents: Optional[int] = Field(default=None, description="Alias for documents count")
    total_chunks: Optional[int] = Field(default=None, description="Alias for chunks count")
    total_entities: Optional[int] = Field(default=None, description="Alias for entities count")
    total_relationships: Optional[int] = Field(default=None, description="Alias for relationships count")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")


# --- Dependencies ---

def get_lancedb_client() -> LanceDBClient:
    """FastAPI dependency to retrieve or initialize LanceDB client."""
    return LanceDBClient()


def get_kuzu_engine() -> KuzuEngine:
    """FastAPI dependency to retrieve or initialize KùzuDB engine."""
    return KuzuEngine()


def get_pg_conn_safe(conn: Optional[Any] = Depends(get_pg_conn)) -> Optional[Any]:
    """Safely pass PostgreSQL connection dependency."""
    return conn


def get_lancedb_client_safe(client: Optional[LanceDBClient] = Depends(get_lancedb_client)) -> Optional[LanceDBClient]:
    """Safely pass LanceDB client dependency."""
    return client


def get_neo4j_client_safe(client: Optional[Neo4jClient] = Depends(get_neo4j_client)) -> Optional[Neo4jClient]:
    """Safely pass Neo4j client dependency."""
    return client


def get_kuzu_engine_safe(engine: Optional[KuzuEngine] = Depends(get_kuzu_engine)) -> Optional[KuzuEngine]:
    """Safely pass KùzuDB engine dependency."""
    return engine


# --- Health Check Helper Functions ---

def check_postgres_health(conn: Any) -> ComponentHealth:
    if conn is None:
        return ComponentHealth(status="unhealthy", details="PostgreSQL connection is unavailable")
    try:
        if hasattr(conn, "cursor"):
            cur = conn.cursor()
            cur.execute("SELECT 1;")
            res = cur.fetchone()
            if hasattr(cur, "close"):
                cur.close()
        else:
            res = conn.execute("SELECT 1;").fetchone()
        return ComponentHealth(status="healthy", details="PostgreSQL connection operational")
    except Exception as e:
        logger.warning("PostgreSQL health check failed: %s", e)
        return ComponentHealth(status="unhealthy", details=f"PostgreSQL check failed: {str(e)}")


def check_lancedb_health(client: Any) -> ComponentHealth:
    if client is None:
        return ComponentHealth(status="unhealthy", details="LanceDB client is unavailable")
    try:
        db = client.connect()
        tables = db.list_tables()
        return ComponentHealth(status="healthy", details="LanceDB connection operational")
    except Exception as e:
        logger.warning("LanceDB health check failed: %s", e)
        return ComponentHealth(status="unhealthy", details=f"LanceDB check failed: {str(e)}")


def check_neo4j_health(client: Any) -> ComponentHealth:
    if client is None:
        return ComponentHealth(status="unhealthy", details="Neo4j client is unavailable")
    try:
        client.execute_query("RETURN 1 AS test")
        return ComponentHealth(status="healthy", details="Neo4j connection operational")
    except Exception as e:
        logger.warning("Neo4j health check failed: %s", e)
        return ComponentHealth(status="unhealthy", details=f"Neo4j check failed: {str(e)}")


def check_kuzu_health(engine: Any) -> ComponentHealth:
    if engine is None:
        return ComponentHealth(status="unhealthy", details="KùzuDB engine is unavailable")
    try:
        if hasattr(engine, "execute_query"):
            engine.execute_query("RETURN 1 AS test")
        return ComponentHealth(status="healthy", details="KùzuDB connection operational")
    except Exception as e:
        logger.warning("KùzuDB health check failed: %s", e)
        return ComponentHealth(status="unhealthy", details=f"KùzuDB check failed: {str(e)}")


# --- Router Endpoints ---

@router.get("/health", response_model=HealthResponse)
def get_health(
    pg_conn: Optional[Any] = Depends(get_pg_conn_safe),
    lancedb_client: Optional[LanceDBClient] = Depends(get_lancedb_client_safe),
    neo4j_client: Optional[Neo4jClient] = Depends(get_neo4j_client_safe),
    kuzu_engine: Optional[KuzuEngine] = Depends(get_kuzu_engine_safe),
) -> HealthResponse:
    """Check connection status for PostgreSQL, LanceDB, Neo4j, and KùzuDB."""
    pg_health = check_postgres_health(pg_conn)
    lancedb_health = check_lancedb_health(lancedb_client)
    neo4j_health = check_neo4j_health(neo4j_client)
    kuzu_health = check_kuzu_health(kuzu_engine)

    components = {
        "postgres": pg_health,
        "lancedb": lancedb_health,
        "neo4j": neo4j_health,
        "kuzu": kuzu_health,
    }

    all_healthy = all(c.status == "healthy" for c in components.values())
    overall_status = "healthy" if all_healthy else "degraded"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        components=components,
    )


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics(
    pg_conn: Any = Depends(get_pg_conn),
) -> MetricsResponse:
    """Query PostgreSQL for total document, chunk, entity, and relationship counts."""
    if pg_conn is None:
        raise HTTPException(status_code=500, detail="PostgreSQL connection is unavailable")

    try:
        query = """
            SELECT 
                (SELECT COUNT(*) FROM documents) AS documents,
                (SELECT COUNT(*) FROM document_chunks) AS chunks,
                (SELECT COUNT(*) FROM normalized_entities) AS entities,
                (SELECT COUNT(*) FROM relationships) AS relationships;
        """
        if hasattr(pg_conn, "cursor"):
            cur = pg_conn.cursor()
            cur.execute(query)
            row = cur.fetchone()
            if hasattr(cur, "close"):
                cur.close()
        else:
            row = pg_conn.execute(query).fetchone()

        if row is None:
            raise HTTPException(status_code=500, detail="Database returned no metric rows")

        if isinstance(row, dict):
            docs = int(row.get("documents", 0))
            chunks = int(row.get("chunks", 0))
            entities = int(row.get("entities", 0))
            rels = int(row.get("relationships", 0))
        elif isinstance(row, (list, tuple)):
            docs = int(row[0])
            chunks = int(row[1])
            entities = int(row[2])
            rels = int(row[3])
        else:
            docs = int(getattr(row, "documents", getattr(row, "total_documents", 0)))
            chunks = int(getattr(row, "chunks", getattr(row, "total_chunks", 0)))
            entities = int(getattr(row, "entities", getattr(row, "total_entities", 0)))
            rels = int(getattr(row, "relationships", getattr(row, "total_relationships", 0)))

        return MetricsResponse(
            documents=docs,
            chunks=chunks,
            entities=entities,
            relationships=rels,
            total_documents=docs,
            total_chunks=chunks,
            total_entities=entities,
            total_relationships=rels,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to query metrics: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to query database metrics: {str(e)}")


health_router = router
