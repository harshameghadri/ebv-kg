"""Neo4j Graph database client wrapper for EBV Knowledge System."""

import os
from typing import Any

from neo4j import GraphDatabase, Driver


class Neo4jClient:
    """Wrapper around the official neo4j Python driver for knowledge graph operations."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        driver: Driver | Any | None = None,
    ) -> None:
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")

        if driver is not None:
            self.driver = driver
        else:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )

    def close(self) -> None:
        """Close the underlying Neo4j driver connection pool."""
        if self.driver is not None:
            self.driver.close()

    def __enter__(self) -> "Neo4jClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _validate_identifier(self, name: str, param_name: str) -> str:
        """Sanitize and validate Cypher identifier to prevent Cypher injection."""
        if not name or not name.isidentifier():
            raise ValueError(f"Invalid Cypher identifier for '{param_name}': '{name}'")
        return name

    def execute_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[Any]:
        """Execute a Cypher query in an auto-closing session and return results."""
        if self.driver is None:
            raise RuntimeError("Neo4j driver is not initialized.")

        parameters = parameters or {}
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters)
            return list(result)

    def init_schema(self) -> list[str]:
        """Initialize graph database schema (unique constraints and indexes).

        Idempotent operation using IF NOT EXISTS.
        Creates unique constraints for:
        - Entity(canonical_id)
        - Paper(doi)
        And indexes for:
        - Entity(name)
        - Entity(entity_type)
        - Paper(pmid)

        Returns:
            List of executed Cypher query strings.
        """
        schema_queries = [
            "CREATE CONSTRAINT entity_canonical_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.canonical_id IS UNIQUE",
            "CREATE CONSTRAINT paper_doi_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.doi IS UNIQUE",
            "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
            "CREATE INDEX paper_pmid_idx IF NOT EXISTS FOR (p:Paper) ON (p.pmid)",
        ]

        for query in schema_queries:
            self.execute_query(query)

        return schema_queries

    def clear_graph(self) -> None:
        """Drop all nodes and relationships from the database (full rebuild)."""
        query = "MATCH (n) DETACH DELETE n"
        self.execute_query(query)

    def bulk_upsert_nodes(
        self,
        label: str,
        nodes: list[dict[str, Any]],
        id_property: str | None = None,
    ) -> int:
        """Bulk upsert nodes into Neo4j using parameterized Cypher and UNWIND.

        Args:
            label: Node label (e.g. 'Entity', 'Paper').
            nodes: List of node property dictionaries.
            id_property: Property key used as unique identifier for MERGE. If None,
                         inferred from label ('canonical_id' for Entity, 'doi' for Paper)
                         or node dict keys.

        Returns:
            Number of nodes processed.
        """
        if not nodes:
            return 0

        label_clean = self._validate_identifier(label, "label")

        if id_property is None:
            if label_clean.lower() in ("paper", "document"):
                id_property = "doi"
            elif label_clean.lower() == "entity":
                id_property = "canonical_id"
            else:
                first_node = nodes[0]
                if "canonical_id" in first_node:
                    id_property = "canonical_id"
                elif "doi" in first_node:
                    id_property = "doi"
                elif "id" in first_node:
                    id_property = "id"
                else:
                    raise ValueError(
                        f"Cannot infer id_property for label '{label_clean}' from node keys: {list(first_node.keys())}"
                    )

        id_prop_clean = self._validate_identifier(id_property, "id_property")

        query = (
            f"UNWIND $nodes AS batch "
            f"MERGE (n:`{label_clean}` {{`{id_prop_clean}`: batch.`{id_prop_clean}`}}) "
            f"SET n += batch"
        )

        self.execute_query(query, {"nodes": nodes})
        return len(nodes)

    def bulk_upsert_edges(
        self,
        rel_type: str,
        edges: list[dict[str, Any]],
        source_label: str = "Entity",
        target_label: str = "Entity",
        source_key: str = "canonical_id",
        target_key: str = "canonical_id",
    ) -> int:
        """Bulk upsert relationships into Neo4j using parameterized Cypher and UNWIND.

        Args:
            rel_type: Relationship type name (e.g. 'TARGETS', 'INTERACTS_WITH', 'MENTIONS').
            edges: List of relationship dictionaries.
            source_label: Label of source nodes.
            target_label: Label of target nodes.
            source_key: Property key matching source nodes.
            target_key: Property key matching target nodes.

        Returns:
            Number of edges processed.
        """
        if not edges:
            return 0

        rel_type_clean = self._validate_identifier(rel_type, "rel_type")
        source_label_clean = self._validate_identifier(source_label, "source_label")
        target_label_clean = self._validate_identifier(target_label, "target_label")
        source_key_clean = self._validate_identifier(source_key, "source_key")
        target_key_clean = self._validate_identifier(target_key, "target_key")

        formatted_edges: list[dict[str, Any]] = []
        for idx, edge in enumerate(edges):
            source_id = (
                edge.get("source_id")
                or edge.get("source_canonical_id")
                or edge.get("source_doi")
                or edge.get("source")
            )
            target_id = (
                edge.get("target_id")
                or edge.get("target_canonical_id")
                or edge.get("target_doi")
                or edge.get("target")
            )

            if source_id is None:
                raise ValueError(
                    f"Edge at index {idx} missing source identifier (expected 'source_id', 'source_canonical_id', 'source_doi', or 'source')"
                )
            if target_id is None:
                raise ValueError(
                    f"Edge at index {idx} missing target identifier (expected 'target_id', 'target_canonical_id', 'target_doi', or 'target')"
                )

            reserved_keys = {
                "source_id",
                "target_id",
                "source_canonical_id",
                "target_canonical_id",
                "source_doi",
                "target_doi",
                "source",
                "target",
                "source_label",
                "target_label",
                "source_key",
                "target_key",
                "properties",
            }
            props = {k: v for k, v in edge.items() if k not in reserved_keys}
            if "properties" in edge and isinstance(edge["properties"], dict):
                props.update(edge["properties"])

            formatted_edges.append(
                {
                    "source_id": str(source_id),
                    "target_id": str(target_id),
                    "properties": props,
                }
            )

        query = (
            f"UNWIND $edges AS batch "
            f"MATCH (source:`{source_label_clean}` {{`{source_key_clean}`: batch.source_id}}) "
            f"MATCH (target:`{target_label_clean}` {{`{target_key_clean}`: batch.target_id}}) "
            f"MERGE (source)-[r:`{rel_type_clean}`]->(target) "
            f"SET r += batch.properties"
        )

        self.execute_query(query, {"edges": formatted_edges})
        return len(edges)
