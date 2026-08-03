"""FastAPI router defining Hypothesis generation and discovery endpoints for the EBV Knowledge System."""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from app.materialization.neo4j_client import Neo4jClient
from app.api.routes import get_neo4j_client, get_pg_conn

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/hypothesis", tags=["hypothesis"])

# --- Pydantic Schemas ---

class NicheOverlapRequest(BaseModel):
    diseases: Optional[List[str]] = Field(
        default=None,
        description="List of disease names or canonical IDs to filter niche overlap across."
    )
    min_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score threshold for entity relationships."
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of niche overlap results to return."
    )
    source: str = Field(
        default="auto",
        description="Data source to query ('auto', 'neo4j', or 'postgres')."
    )

class DiseaseOutcomeInfo(BaseModel):
    id: Optional[str] = Field(default=None, description="Canonical ID of the disease outcome entity")
    name: str = Field(..., description="Disease outcome name")
    confidence_score: float = Field(default=1.0, description="Confidence score of connection")

class MarkerGeneInfo(BaseModel):
    id: Optional[str] = Field(default=None, description="Canonical ID of the gene entity")
    name: str = Field(..., description="Gene name or symbol")
    symbol: Optional[str] = Field(default=None, description="Gene symbol")

class NicheOverlapItem(BaseModel):
    cell_state_id: Optional[str] = Field(default=None, description="Canonical ID of the cell state entity")
    cell_state_name: str = Field(..., description="Name of shared cell state (e.g. Atypical B Cell)")
    connected_diseases: List[DiseaseOutcomeInfo] = Field(
        default_factory=list,
        description="Disease outcome entities connected across silos"
    )
    marker_genes: List[MarkerGeneInfo] = Field(
        default_factory=list,
        description="Marker genes associated with this cell state"
    )
    overlap_confidence: float = Field(
        ...,
        description="Aggregate confidence score for the niche overlap hypothesis"
    )
    silo_count: int = Field(
        ...,
        description="Number of distinct disease outcome silos connected to this cell state"
    )

class NicheOverlapResponse(BaseModel):
    total_results: int = Field(..., description="Total number of overlapping cell states found")
    overlaps: List[NicheOverlapItem] = Field(..., description="List of niche overlap hypothesis items")



def _get_record_val(rec: Any, key: str, default: Any = None) -> Any:
    """Helper to extract a dictionary or record attribute safely."""
    if hasattr(rec, "get"):
        v = rec.get(key)
        if v is not None:
            return v
    if isinstance(rec, dict):
        return rec.get(key, default)
    try:
        return rec[key]
    except Exception:
        return default


def _query_neo4j_niche_overlap(
    neo4j_client: Neo4jClient,
    diseases: Optional[List[str]],
    min_confidence: float,
    limit: int
) -> List[NicheOverlapItem]:
    """Execute Cypher query against Neo4j to discover CellState nodes connected to >= 2 DiseaseOutcomes."""
    cypher = """
    MATCH (cs:Entity)
    WHERE cs.entity_type IN ['CELL_STATE', 'CellState', 'CELL_TYPE'] OR 'CellState' IN labels(cs)
    MATCH (cs)-[r1]-(d:Entity)
    WHERE (d.entity_type IN ['DISEASE', 'DISEASE_OUTCOME', 'DiseaseOutcome', 'DISEASE_STATUS'] OR 'DiseaseOutcome' IN labels(d) OR 'Disease' IN labels(d))
      AND (r1.confidence_score IS NULL OR r1.confidence_score >= $min_confidence)
    """

    params: Dict[str, Any] = {
        "min_confidence": min_confidence,
        "limit": limit
    }

    if diseases:
        cypher += """
        AND (ANY(dis IN $diseases WHERE toLower(d.name) CONTAINS toLower(dis) OR toLower(dis) CONTAINS toLower(d.name) OR d.canonical_id = dis))
        """
        params["diseases"] = diseases

    cypher += """
    OPTIONAL MATCH (cs)-[r2]-(g:Entity)
    WHERE g.entity_type IN ['GENE', 'MARKER', 'Gene'] OR 'Gene' IN labels(g)
    WITH cs, d, r1, g
    WITH cs,
         collect(DISTINCT {
             id: d.canonical_id,
             name: d.name,
             confidence_score: coalesce(r1.confidence_score, 1.0)
         }) AS raw_diseases,
         collect(DISTINCT {
             id: g.canonical_id,
             name: g.name,
             symbol: g.name
         }) AS raw_genes,
         avg(coalesce(r1.confidence_score, 1.0)) AS avg_conf
    WHERE size(raw_diseases) >= 2
    RETURN cs.canonical_id AS cell_state_id,
           cs.name AS cell_state_name,
           raw_diseases,
           raw_genes,
           avg_conf AS overlap_confidence,
           size(raw_diseases) AS silo_count
    ORDER BY overlap_confidence DESC, silo_count DESC
    LIMIT $limit
    """

    results = neo4j_client.execute_query(cypher, params)
    items: List[NicheOverlapItem] = []

    for rec in results:
        cs_id = _get_record_val(rec, "cell_state_id")
        cs_name = _get_record_val(rec, "cell_state_name") or cs_id or "Unknown Cell State"
        raw_diseases = _get_record_val(rec, "raw_diseases") or []
        raw_genes = _get_record_val(rec, "raw_genes") or []
        overlap_conf = float(_get_record_val(rec, "overlap_confidence") or 0.0)

        connected_diseases: List[DiseaseOutcomeInfo] = []
        seen_d = set()
        for d in raw_diseases:
            if isinstance(d, dict) and d.get("name"):
                d_name = d["name"]
                if d_name not in seen_d:
                    seen_d.add(d_name)
                    connected_diseases.append(
                        DiseaseOutcomeInfo(
                            id=d.get("id"),
                            name=d_name,
                            confidence_score=float(d.get("confidence_score", 1.0))
                        )
                    )

        marker_genes: List[MarkerGeneInfo] = []
        seen_g = set()
        for g in raw_genes:
            if isinstance(g, dict) and g.get("name"):
                g_name = g["name"]
                if g_name not in seen_g:
                    seen_g.add(g_name)
                    marker_genes.append(
                        MarkerGeneInfo(
                            id=g.get("id"),
                            name=g_name,
                            symbol=g.get("symbol") or g_name
                        )
                    )

        if len(connected_diseases) >= 2:
            items.append(
                NicheOverlapItem(
                    cell_state_id=cs_id,
                    cell_state_name=cs_name,
                    connected_diseases=connected_diseases,
                    marker_genes=marker_genes,
                    overlap_confidence=round(overlap_conf, 4),
                    silo_count=len(connected_diseases)
                )
            )

    return items


def _query_postgres_niche_overlap(
    conn: Any,
    diseases: Optional[List[str]],
    min_confidence: float,
    limit: int
) -> List[NicheOverlapItem]:
    """Execute SQL queries against PostgreSQL to discover CellState nodes connected to >= 2 DiseaseOutcomes."""
    from psycopg.rows import dict_row

    sql_cs = """
    SELECT 
        cs.id AS internal_cs_id,
        cs.canonical_id AS cell_state_id,
        cs.name AS cell_state_name
    FROM normalized_entities cs
    JOIN relationships r ON (r.source_entity_id = cs.id OR r.target_entity_id = cs.id)
    JOIN normalized_entities d ON (
        (r.source_entity_id = d.id AND r.target_entity_id = cs.id) OR
        (r.target_entity_id = d.id AND r.source_entity_id = cs.id)
    )
    WHERE (cs.entity_type ILIKE '%%CELL_STATE%%' OR cs.entity_type ILIKE '%%CELL_TYPE%%' OR cs.entity_type ILIKE '%%CELL%%')
      AND (d.entity_type ILIKE '%%DISEASE%%' OR d.entity_type ILIKE '%%OUTCOME%%' OR d.entity_type ILIKE '%%PATHOLOGY%%')
      AND (r.confidence_score IS NULL OR r.confidence_score >= %s)
    GROUP BY cs.id, cs.canonical_id, cs.name
    HAVING COUNT(DISTINCT d.id) >= 2
    LIMIT %s;
    """

    items: List[NicheOverlapItem] = []
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql_cs, (min_confidence, limit * 2))
        cs_rows = cur.fetchall()

        for cs in cs_rows:
            cs_internal_id = cs["internal_cs_id"]
            cs_id = cs["cell_state_id"]
            cs_name = cs["cell_state_name"] or cs_id or "Unknown Cell State"

            # Query connected diseases
            sql_d = """
            SELECT DISTINCT
                d.canonical_id AS id,
                d.name AS name,
                COALESCE(r.confidence_score, 1.0) AS confidence_score
            FROM relationships r
            JOIN normalized_entities d ON (
                (r.source_entity_id = d.id AND r.target_entity_id = %s) OR
                (r.target_entity_id = d.id AND r.source_entity_id = %s)
            )
            WHERE (d.entity_type ILIKE '%%DISEASE%%' OR d.entity_type ILIKE '%%OUTCOME%%' OR d.entity_type ILIKE '%%PATHOLOGY%%')
              AND (r.confidence_score IS NULL OR r.confidence_score >= %s)
            """
            cur.execute(sql_d, (cs_internal_id, cs_internal_id, min_confidence))
            d_rows = cur.fetchall()

            connected_diseases = []
            seen_d = set()
            for row in d_rows:
                d_name = row["name"]
                if d_name and d_name not in seen_d:
                    seen_d.add(d_name)
                    connected_diseases.append(
                        DiseaseOutcomeInfo(
                            id=row.get("id"),
                            name=d_name,
                            confidence_score=float(row.get("confidence_score", 1.0))
                        )
                    )

            if len(connected_diseases) < 2:
                continue

            # Disease filter check if specified
            if diseases:
                disease_matches = 0
                for requested_d in diseases:
                    req_lower = requested_d.lower()
                    if any(
                        req_lower in d.name.lower()
                        or d.name.lower() in req_lower
                        or (d.id and d.id == requested_d)
                        for d in connected_diseases
                    ):
                        disease_matches += 1
                if disease_matches < 2 and len(diseases) >= 2:
                    continue

            # Query connected marker genes
            sql_g = """
            SELECT DISTINCT
                g.canonical_id AS id,
                g.name AS name
            FROM relationships r
            JOIN normalized_entities g ON (
                (r.source_entity_id = g.id AND r.target_entity_id = %s) OR
                (r.target_entity_id = g.id AND r.source_entity_id = %s)
            )
            WHERE (g.entity_type ILIKE '%%GENE%%' OR g.entity_type ILIKE '%%MARKER%%')
            """
            cur.execute(sql_g, (cs_internal_id, cs_internal_id))
            g_rows = cur.fetchall()

            marker_genes = []
            seen_g = set()
            for row in g_rows:
                g_name = row["name"]
                if g_name and g_name not in seen_g:
                    seen_g.add(g_name)
                    marker_genes.append(
                        MarkerGeneInfo(
                            id=row.get("id"),
                            name=g_name,
                            symbol=g_name
                        )
                    )

            overlap_conf = sum(d.confidence_score for d in connected_diseases) / len(connected_diseases)

            items.append(
                NicheOverlapItem(
                    cell_state_id=cs_id,
                    cell_state_name=cs_name,
                    connected_diseases=connected_diseases,
                    marker_genes=marker_genes,
                    overlap_confidence=round(overlap_conf, 4),
                    silo_count=len(connected_diseases)
                )
            )

            if len(items) >= limit:
                break

    items.sort(key=lambda x: (x.overlap_confidence, x.silo_count), reverse=True)
    return items


def _fetch_niche_overlaps(
    diseases: Optional[List[str]],
    min_confidence: float,
    limit: int,
    source: str,
    neo4j_client: Optional[Neo4jClient],
    pg_conn: Any,
) -> NicheOverlapResponse:
    source_clean = (source or "auto").lower()
    items: List[NicheOverlapItem] = []

    # 1. Try Neo4j if source is 'auto' or 'neo4j'
    if source_clean in ("auto", "neo4j") and neo4j_client is not None:
        try:
            items = _query_neo4j_niche_overlap(neo4j_client, diseases, min_confidence, limit)
        except Exception as e:
            logger.warning("Neo4j niche overlap query failed: %s", e)
            if source_clean == "neo4j":
                raise HTTPException(status_code=500, detail=f"Neo4j query error: {str(e)}")

    # 2. Fallback to or explicitly use PostgreSQL if items is empty or source is 'postgres'
    if not items and source_clean in ("auto", "postgres") and pg_conn is not None:
        try:
            items = _query_postgres_niche_overlap(pg_conn, diseases, min_confidence, limit)
        except Exception as e:
            logger.warning("PostgreSQL niche overlap query failed: %s", e)
            if source_clean == "postgres":
                raise HTTPException(status_code=500, detail=f"PostgreSQL query error: {str(e)}")

    return NicheOverlapResponse(
        total_results=len(items),
        overlaps=items
    )


# --- Endpoints ---

@router.get("/niche-overlap", response_model=NicheOverlapResponse)
async def get_niche_overlap(
    diseases: Optional[List[str]] = Query(
        None,
        description="Filter by disease outcome names or canonical IDs (can be repeated or comma-separated)"
    ),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence score threshold"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results limit"),
    source: str = Query("auto", description="Query source ('auto', 'neo4j', or 'postgres')"),
    neo4j_client: Optional[Neo4jClient] = Depends(get_neo4j_client),
    pg_conn = Depends(get_pg_conn),
):
    """Retrieve shared cell state nodes connected to multiple distinct disease outcomes across silos."""
    parsed_diseases: Optional[List[str]] = None
    if diseases:
        parsed_diseases = []
        for d in diseases:
            for part in d.split(","):
                part_clean = part.strip()
                if part_clean:
                    parsed_diseases.append(part_clean)
        if not parsed_diseases:
            parsed_diseases = None

    return _fetch_niche_overlaps(
        diseases=parsed_diseases,
        min_confidence=min_confidence,
        limit=limit,
        source=source,
        neo4j_client=neo4j_client,
        pg_conn=pg_conn,
    )


@router.post("/niche-overlap", response_model=NicheOverlapResponse)
async def post_niche_overlap(
    req: NicheOverlapRequest,
    neo4j_client: Optional[Neo4jClient] = Depends(get_neo4j_client),
    pg_conn = Depends(get_pg_conn),
):
    """POST endpoint to retrieve shared cell state nodes connected to multiple distinct disease outcomes across silos."""
    return _fetch_niche_overlaps(
        diseases=req.diseases,
        min_confidence=req.min_confidence,
        limit=req.limit,
        source=req.source,
        neo4j_client=neo4j_client,
        pg_conn=pg_conn,
    )
