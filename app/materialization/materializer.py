"""Materialization Pipeline for syncing data from PostgreSQL to Neo4j."""

import argparse
import logging
import os
import sys
from collections import defaultdict
from typing import Any

import psycopg
from psycopg.rows import dict_row

from app.materialization.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class Materializer:
    """Handles materializing entity and relationship data from PostgreSQL to Neo4j."""

    def __init__(self, neo4j_client: Neo4jClient | None = None) -> None:
        """Initialize the Materializer.

        Args:
            neo4j_client: An instance of Neo4jClient. If None, a new one is created.
        """
        self.neo4j_client = neo4j_client or Neo4jClient()

    def init_schema(self) -> list[str]:
        """Initialize the Neo4j database schema (constraints and indexes).

        Returns:
            List of executed Cypher query strings.
        """
        logger.info("Initializing Neo4j schema constraints and indexes...")
        return self.neo4j_client.init_schema()

    def clear_graph(self) -> None:
        """Clear all nodes and relationships from the Neo4j database."""
        logger.info("Clearing all data from Neo4j graph...")
        self.neo4j_client.clear_graph()

    def materialize_graph(
        self, pg_conn: Any, curation_statuses: list[str] | None = None
    ) -> dict[str, int]:
        """Materializes nodes and edges from PostgreSQL to Neo4j.

        Args:
            pg_conn: A psycopg Connection object.
            curation_statuses: If specified, only sync relationships
                               (and their mentions) with these curation
                               statuses. E.g., ['APPROVED'].

        Returns:
            A dictionary containing counts of upserted entities, papers, relationships,
            and paper-to-entity mentions.
        """
        logger.info("Starting graph materialization...")
        stats = {
            "entities": 0,
            "papers": 0,
            "relationships": 0,
            "mentions": 0,
        }

        # 1. Fetch and upsert Entity nodes
        logger.info("Fetching entities from PostgreSQL...")
        with pg_conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT id, canonical_id, name, entity_type, ontology_source, synonyms "
                "FROM normalized_entities"
            )
            entities = cur.fetchall()

        if entities:
            entity_nodes = []
            for ent in entities:
                entity_nodes.append(
                    {
                        "id": str(ent["id"]),
                        "canonical_id": ent["canonical_id"],
                        "name": ent["name"],
                        "entity_type": ent["entity_type"],
                        "ontology_source": ent["ontology_source"],
                        "synonyms": ent["synonyms"]
                        if ent["synonyms"] is not None
                        else [],
                    }
                )
            logger.info(f"Upserting {len(entity_nodes)} Entity nodes to Neo4j...")
            stats["entities"] = self.neo4j_client.bulk_upsert_nodes(
                label="Entity",
                nodes=entity_nodes,
                id_property="canonical_id",
            )

        # 2. Fetch and upsert Paper nodes
        logger.info("Fetching papers from PostgreSQL...")
        with pg_conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT id, doi, pmid, title, journal, published_date FROM documents"
            )
            papers = cur.fetchall()

        if papers:
            paper_nodes = []
            for p in papers:
                if not p["doi"]:
                    logger.warning(
                        f"Skipping paper with ID {p['id']} because it lacks a DOI."
                    )
                    continue
                paper_nodes.append(
                    {
                        "id": str(p["id"]),
                        "doi": p["doi"],
                        "pmid": p["pmid"],
                        "title": p["title"],
                        "journal": p["journal"],
                        "published_date": (
                            p["published_date"].isoformat()
                            if p["published_date"] is not None
                            else None
                        ),
                    }
                )
            if paper_nodes:
                logger.info(f"Upserting {len(paper_nodes)} Paper nodes to Neo4j...")
                stats["papers"] = self.neo4j_client.bulk_upsert_nodes(
                    label="Paper",
                    nodes=paper_nodes,
                    id_property="doi",
                )

        # 3. Fetch and upsert relationships (grouped by relationship_type) with rich edge metadata
        logger.info("Fetching relationships with rich evidence properties from PostgreSQL...")
        rel_query = (
            "SELECT "
            "  r.id, "
            "  r.relationship_type, "
            "  r.confidence_score, "
            "  r.curation_status, "
            "  r.source_type, "
            "  src.canonical_id AS source_canonical_id, "
            "  tgt.canonical_id AS target_canonical_id, "
            "  ARRAY_AGG(DISTINCT d.pmid) FILTER (WHERE d.pmid IS NOT NULL) AS source_pmids, "
            "  ARRAY_AGG(DISTINCT d.doi) FILTER (WHERE d.doi IS NOT NULL) AS source_dois, "
            "  COUNT(DISTINCT ev.id) AS evidence_count "
            "FROM relationships r "
            "JOIN normalized_entities src ON r.source_entity_id = src.id "
            "JOIN normalized_entities tgt ON r.target_entity_id = tgt.id "
            "LEFT JOIN relationship_evidence ev ON r.id = ev.relationship_id "
            "LEFT JOIN document_chunks c ON ev.chunk_id = c.id "
            "LEFT JOIN documents d ON c.document_id = d.id "
        )
        rel_params = []
        if curation_statuses is not None:
            rel_query += " WHERE r.curation_status = ANY(%s)"
            rel_params.append(list(curation_statuses))

        rel_query += (
            " GROUP BY r.id, r.relationship_type, r.confidence_score, r.curation_status, "
            "r.source_type, src.canonical_id, tgt.canonical_id"
        )

        with pg_conn.cursor(row_factory=dict_row) as cur:
            cur.execute(rel_query, rel_params)
            relationships = cur.fetchall()

        if relationships:
            edges_by_type = defaultdict(list)
            for row in relationships:
                ev_count = row.get("evidence_count") or 0
                source_type = row.get("source_type") or "NER"
                ev_tier = "DIRECT_LITERATURE_EVIDENCE" if ev_count > 0 else (
                    "SINGLE_CELL_ASSAY" if source_type == "single_cell" else "INFERRED"
                )
                edge_dict = {
                    "id": str(row["id"]),
                    "source_canonical_id": row["source_canonical_id"],
                    "target_canonical_id": row["target_canonical_id"],
                    "confidence_score": row.get("confidence_score", 1.0),
                    "curation_status": row.get("curation_status", "APPROVED"),
                    "source_type": source_type,
                    "evidence_count": ev_count,
                    "evidence_tier": ev_tier,
                    "source_pmids": row.get("source_pmids") or [],
                    "source_dois": row.get("source_dois") or [],
                }
                edges_by_type[row["relationship_type"]].append(edge_dict)


            total_edges = 0
            for rel_type, edges in edges_by_type.items():
                logger.info(f"Upserting {len(edges)} '{rel_type}' edges with rich properties to Neo4j...")
                upserted = self.neo4j_client.bulk_upsert_edges(
                    rel_type=rel_type,
                    edges=edges,
                    source_label="Entity",
                    target_label="Entity",
                    source_key="canonical_id",
                    target_key="canonical_id",
                )
                total_edges += upserted
            stats["relationships"] = total_edges


        # 4. Draw 'MENTIONS' edges between Paper and Entity nodes
        logger.info("Fetching entity evidence mentions from PostgreSQL...")
        mentions_query = (
            "SELECT "
            "  d.doi AS source_doi, "
            "  ent.canonical_id AS target_canonical_id, "
            "  MAX(ev.confidence_score) AS confidence_score "
            "FROM documents d "
            "JOIN document_chunks c ON d.id = c.document_id "
            "JOIN relationship_evidence ev ON c.id = ev.chunk_id "
            "JOIN relationships r ON ev.relationship_id = r.id "
            "JOIN normalized_entities ent ON "
            "  (ent.id = r.source_entity_id OR ent.id = r.target_entity_id)"
        )
        mentions_params = []
        if curation_statuses is not None:
            mentions_query += " WHERE r.curation_status = ANY(%s)"
            mentions_params.append(list(curation_statuses))

        mentions_query += " GROUP BY d.doi, ent.canonical_id"

        with pg_conn.cursor(row_factory=dict_row) as cur:
            cur.execute(mentions_query, mentions_params)
            mentions = cur.fetchall()

        if mentions:
            mentions_edges = []
            for row in mentions:
                if not row["source_doi"] or not row["target_canonical_id"]:
                    continue
                mentions_edges.append(
                    {
                        "source_doi": row["source_doi"],
                        "target_canonical_id": row["target_canonical_id"],
                        "confidence_score": row["confidence_score"],
                    }
                )

            if mentions_edges:
                logger.info(
                    f"Upserting {len(mentions_edges)} 'MENTIONS' edges to Neo4j..."
                )
                stats["mentions"] = self.neo4j_client.bulk_upsert_edges(
                    rel_type="MENTIONS",
                    edges=mentions_edges,
                    source_label="Paper",
                    target_label="Entity",
                    source_key="doi",
                    target_key="canonical_id",
                )

        logger.info(
            f"Graph materialization complete: "
            f"{stats['entities']} entities, {stats['papers']} papers, "
            f"{stats['relationships']} relationships, {stats['mentions']} mentions."
        )
        return stats


def main() -> None:
    """CLI entrypoint for graph materialization."""
    parser = argparse.ArgumentParser(
        description=(
            "Materialize Epstein-Barr Virus Knowledge Graph from PostgreSQL to Neo4j."
        )
    )
    parser.add_argument(
        "--pg-dsn",
        default=os.getenv("DATABASE_URL") or os.getenv("POSTGRES_DSN"),
        help="PostgreSQL connection DSN / connection string.",
    )
    parser.add_argument(
        "--neo4j-uri",
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j URI.",
    )
    parser.add_argument(
        "--neo4j-user",
        default=os.getenv("NEO4J_USER", "neo4j"),
        help="Neo4j username.",
    )
    parser.add_argument(
        "--neo4j-password",
        default=os.getenv("NEO4J_PASSWORD", "password"),
        help="Neo4j password.",
    )
    parser.add_argument(
        "--neo4j-database",
        default=os.getenv("NEO4J_DATABASE", "neo4j"),
        help="Neo4j database name.",
    )
    parser.add_argument(
        "--curation-status",
        action="append",
        help=(
            "Curation status to filter relationships by (e.g. APPROVED, PENDING). "
            "Can be specified multiple times. If not specified, "
            "all relationships are materialized."
        ),
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the existing Neo4j graph before materializing.",
    )
    parser.add_argument(
        "--init-schema",
        action="store_true",
        help="Initialize constraints and indexes in Neo4j before materializing.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose log output.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not args.pg_dsn:
        logger.error(
            "PostgreSQL DSN must be provided via --pg-dsn argument, "
            "or DATABASE_URL/POSTGRES_DSN environment variables."
        )
        sys.exit(1)

    logger.info("Connecting to Neo4j...")
    try:
        neo4j_client = Neo4jClient(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password,
            database=args.neo4j_database,
        )
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        sys.exit(1)

    logger.info("Connecting to PostgreSQL...")
    try:
        pg_conn = psycopg.connect(args.pg_dsn)
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        neo4j_client.close()
        sys.exit(1)

    try:
        materializer = Materializer(neo4j_client=neo4j_client)

        if args.clear:
            materializer.clear_graph()

        if args.init_schema:
            materializer.init_schema()

        materializer.materialize_graph(
            pg_conn=pg_conn,
            curation_statuses=args.curation_status,
        )
    except Exception as e:
        logger.error(f"Error during materialization: {e}")
        sys.exit(1)
    finally:
        pg_conn.close()
        neo4j_client.close()


if __name__ == "__main__":
    main()
