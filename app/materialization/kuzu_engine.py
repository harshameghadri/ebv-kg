"""Kùzu Embedded Graph Database Engine Wrapper for EBV Knowledge System.

Provides high-performance, embedded C++ graph storage using Kùzu with Cypher query support,
schema management, bulk parameterized write operations, and 2-hop neighborhood path retrieval.
Includes robust import handling with a clean mock fallback if kuzu is unavailable.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Attempt to import kuzu safely
try:
    import kuzu
    KUZU_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    kuzu = None
    KUZU_AVAILABLE = False


class MockQueryResult:
    """Mock QueryResult class for fallback mode."""

    def __init__(
        self,
        rows: Optional[List[Dict[str, Any]]] = None,
        column_names: Optional[List[str]] = None,
    ) -> None:
        self._rows = rows or []
        if column_names is not None:
            self._column_names = column_names
        elif self._rows:
            self._column_names = list(self._rows[0].keys())
        else:
            self._column_names = []
        self._index = 0

    def get_column_names(self) -> List[str]:
        return self._column_names

    def has_next(self) -> bool:
        return self._index < len(self._rows)

    def get_next(self) -> List[Any]:
        if not self.has_next():
            raise StopIteration("No more rows in MockQueryResult")
        row = self._rows[self._index]
        self._index += 1
        return [row.get(col) for col in self._column_names]

    def rows_as_dict(self) -> Any:
        return iter(self._rows)

    def get_as_df(self) -> Any:
        try:
            import pandas as pd
            return pd.DataFrame(self._rows)
        except Exception:
            return self._rows

    def close(self) -> None:
        pass


class MockKuzuDatabase:
    """Mock Database class for fallback mode."""

    def __init__(
        self, db_path: str = ":memory:", buffer_pool_size: int = 1024 * 1024 * 1024
    ) -> None:
        self.db_path = db_path
        self.buffer_pool_size = buffer_pool_size
        self.is_closed = False
        self.node_tables: Dict[str, Dict[str, Any]] = {}
        self.rel_tables: Dict[str, Dict[str, Any]] = {}
        self.nodes: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.relationships: Dict[str, List[Dict[str, Any]]] = {}

    def close(self) -> None:
        self.is_closed = True


class MockKuzuConnection:
    """Mock Connection class for fallback mode."""

    def __init__(self, db: MockKuzuDatabase) -> None:
        self.db = db
        self.is_closed = False

    def close(self) -> None:
        self.is_closed = True

    def execute(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> MockQueryResult:
        parameters = parameters or {}
        q_strip = query.strip()
        q_upper = q_strip.upper()

        if "CREATE NODE TABLE" in q_upper:
            clean_q = query.replace("IF NOT EXISTS", "")
            tokens = clean_q.split()
            table_token = "TABLE"
            idx = tokens.index(table_token) + 1 if table_token in tokens else 3
            label = tokens[idx].split("(")[0].strip()
            self.db.node_tables[label] = {}
            if label not in self.db.nodes:
                self.db.nodes[label] = {}
            return MockQueryResult([])

        if "CREATE REL TABLE" in q_upper:
            clean_q = query.replace("IF NOT EXISTS", "")
            tokens = clean_q.split()
            table_token = "TABLE"
            idx = tokens.index(table_token) + 1 if table_token in tokens else 3
            rel_type = tokens[idx].split("(")[0].strip()
            self.db.rel_tables[rel_type] = {}
            if rel_type not in self.db.relationships:
                self.db.relationships[rel_type] = []
            return MockQueryResult([])

        if "UNWIND" in q_upper and "MERGE" in q_upper:
            if "nodes" in parameters:
                node_list = parameters["nodes"]
                label = "Entity"
                for l in self.db.node_tables:
                    if f":{l}" in query or f":`{l}`" in query:
                        label = l
                        break
                if label not in self.db.nodes:
                    self.db.nodes[label] = {}

                id_prop = (
                    "canonical_id"
                    if label == "Entity"
                    else ("doi" if label == "Paper" else "id")
                )
                for n in node_list:
                    key_val = n.get(
                        id_prop, n.get("canonical_id", n.get("doi", n.get("id")))
                    )
                    if key_val:
                        if key_val not in self.db.nodes[label]:
                            self.db.nodes[label][key_val] = {}
                        self.db.nodes[label][key_val].update(n)
                return MockQueryResult([])

            elif "edges" in parameters or "relationships" in parameters:
                edge_list = (
                    parameters.get("edges") or parameters.get("relationships") or []
                )
                rel_type = "ASSOCIATED_WITH"
                for r in self.db.rel_tables:
                    if f":{r}" in query or f":`{r}`" in query:
                        rel_type = r
                        break
                if rel_type not in self.db.relationships:
                    self.db.relationships[rel_type] = []

                for edge in edge_list:
                    self.db.relationships[rel_type].append(edge)
                return MockQueryResult([])

        if "DELETE" in q_upper:
            if "DELETE R" in q_upper:
                self.db.relationships = {r: [] for r in self.db.rel_tables}
            if "DELETE N" in q_upper:
                self.db.nodes = {lbl: {} for lbl in self.db.node_tables}
            return MockQueryResult([])

        if "MATCH" in q_upper:
            entity_id = parameters.get("entity_id") or parameters.get("id")
            if not entity_id and "entity_ids" in parameters:
                eids = parameters["entity_ids"]
                entity_id = eids[0] if isinstance(eids, list) and eids else eids

            if (
                "PATH" in q_upper
                or "HOP" in q_upper
                or "*1..2" in q_upper
                or "HOP1" in q_upper
                or "HOP2" in q_upper
                or ("-[R1]" in q_upper and "-[R2]" in q_upper)
            ):
                rows = []
                if entity_id:
                    start_node = self.db.nodes.get("Entity", {}).get(
                        entity_id, {"canonical_id": entity_id, "name": entity_id}
                    )
                    hop1_edges = []
                    for rel_type, rel_list in self.db.relationships.items():
                        for edge in rel_list:
                            s_id = (
                                edge.get("source_id")
                                or edge.get("source_canonical_id")
                                or edge.get("source")
                            )
                            t_id = (
                                edge.get("target_id")
                                or edge.get("target_canonical_id")
                                or edge.get("target")
                            )
                            if s_id == entity_id or t_id == entity_id:
                                next_id = t_id if s_id == entity_id else s_id
                                hop1_edges.append((edge, rel_type, next_id))

                    for edge1, rel_type1, hop1_id in hop1_edges:
                        hop2_found = False
                        for rel_type2, rel_list2 in self.db.relationships.items():
                            for edge2 in rel_list2:
                                s_id2 = (
                                    edge2.get("source_id")
                                    or edge2.get("source_canonical_id")
                                    or edge2.get("source")
                                )
                                t_id2 = (
                                    edge2.get("target_id")
                                    or edge2.get("target_canonical_id")
                                    or edge2.get("target")
                                )
                                other_id = (
                                    t_id2
                                    if s_id2 == hop1_id
                                    else (s_id2 if t_id2 == hop1_id else None)
                                )
                                if other_id and other_id != entity_id:
                                    hop2_found = True
                                    hop1_node = self.db.nodes.get("Entity", {}).get(
                                        hop1_id,
                                        {"canonical_id": hop1_id, "name": hop1_id},
                                    )
                                    hop2_node = self.db.nodes.get("Entity", {}).get(
                                        other_id,
                                        {"canonical_id": other_id, "name": other_id},
                                    )
                                    rows.append(
                                        {
                                            "start_id": entity_id,
                                            "hop1_id": hop1_id,
                                            "hop1_name": hop1_node.get("name", hop1_id),
                                            "rel1_type": rel_type1,
                                            "hop2_id": other_id,
                                            "hop2_name": hop2_node.get("name", other_id),
                                            "rel2_type": rel_type2,
                                            "path": {
                                                "_nodes": [
                                                    start_node,
                                                    hop1_node,
                                                    hop2_node,
                                                ],
                                                "_rels": [edge1, edge2],
                                            },
                                        }
                                    )
                        if not hop2_found:
                            hop1_node = self.db.nodes.get("Entity", {}).get(
                                hop1_id, {"canonical_id": hop1_id, "name": hop1_id}
                            )
                            rows.append(
                                {
                                    "start_id": entity_id,
                                    "hop1_id": hop1_id,
                                    "hop1_name": hop1_node.get("name", hop1_id),
                                    "rel1_type": rel_type1,
                                    "hop2_id": None,
                                    "hop2_name": None,
                                    "rel2_type": None,
                                    "path": {
                                        "_nodes": [start_node, hop1_node],
                                        "_rels": [edge1],
                                    },
                                }
                            )
                return MockQueryResult(rows)

            rows = []
            for label, node_dict in self.db.nodes.items():
                for node_id, node_data in node_dict.items():
                    if (
                        entity_id is None
                        or node_id == entity_id
                        or node_data.get("name") == entity_id
                    ):
                        row = dict(node_data)
                        if "AS ID" in q_upper:
                            row["id"] = (
                                node_data.get("canonical_id")
                                or node_data.get("doi")
                                or node_data.get("id")
                            )
                        rows.append(row)
            return MockQueryResult(rows)

        return MockQueryResult([])


class KuzuEngine:
    """Wrapper around Kùzu embedded C++ graph database with schema and bulk execution support."""

    def __init__(
        self,
        db_path: str = ":memory:",
        buffer_pool_size: int = 1024 * 1024 * 1024,
        force_mock: bool = False,
    ) -> None:
        """Initialize KuzuEngine.

        Args:
            db_path: Path to database directory, or ':memory:' for transient in-memory DB.
            buffer_pool_size: Maximum buffer pool size in bytes (default 1GB).
            force_mock: If True, forces use of MockKuzu engine regardless of kuzu availability.
        """
        self.db_path = db_path
        self.buffer_pool_size = buffer_pool_size
        self.is_mock = force_mock or not KUZU_AVAILABLE

        if not self.is_mock and kuzu is not None:
            logger.info("Initializing native C++ Kùzu database at '%s'", db_path)
            self.db = kuzu.Database(db_path, buffer_pool_size=buffer_pool_size)
            self.conn = kuzu.Connection(self.db)
        else:
            logger.info("Initializing fallback Mock Kùzu engine")
            self.db = MockKuzuDatabase(db_path, buffer_pool_size=buffer_pool_size)
            self.conn = MockKuzuConnection(self.db)

    def close(self) -> None:
        """Close the underlying connection and database."""
        if hasattr(self, "conn") and self.conn is not None:
            try:
                self.conn.close()
            except Exception as e:
                logger.warning("Error closing Kùzu connection: %s", e)
        if hasattr(self, "db") and self.db is not None:
            try:
                self.db.close()
            except Exception as e:
                logger.warning("Error closing Kùzu database: %s", e)

    def __enter__(self) -> "KuzuEngine":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def execute_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query with parameters and return list of result dictionaries."""
        if self.conn is None:
            raise RuntimeError("Kuzu connection is not initialized.")
        parameters = parameters or {}
        result = self.conn.execute(query, parameters)
        if hasattr(result, "rows_as_dict"):
            return list(result.rows_as_dict())
        return []

    def init_schema(self) -> List[str]:
        """Initialize Kùzu node and relationship schemas.

        Node tables created:
        - Entity(canonical_id STRING PRIMARY KEY, name STRING, entity_type STRING, ontology_source STRING, synonyms STRING[])
        - Paper(doi STRING PRIMARY KEY, pmid STRING, title STRING, journal STRING, publication_year INT64)

        Rel tables created:
        - ASSOCIATED_WITH(FROM Entity TO Entity, FROM Paper TO Entity, FROM Entity TO Paper, relationship_type STRING, confidence DOUBLE, evidence_text STRING, curation_status STRING)
        - IS_MARKER_FOR(FROM Entity TO Entity, log2_fold_change DOUBLE, p_value DOUBLE, cell_type STRING, confidence DOUBLE)

        Returns:
            List of executed DDL Cypher queries.
        """
        schema_queries = [
            "CREATE NODE TABLE IF NOT EXISTS Entity(canonical_id STRING, name STRING, entity_type STRING, ontology_source STRING, synonyms STRING[], PRIMARY KEY(canonical_id));",
            "CREATE NODE TABLE IF NOT EXISTS Paper(doi STRING, pmid STRING, title STRING, journal STRING, publication_year INT64, PRIMARY KEY(doi));",
            "CREATE REL TABLE IF NOT EXISTS ASSOCIATED_WITH(FROM Entity TO Entity, FROM Paper TO Entity, FROM Entity TO Paper, relationship_type STRING, confidence DOUBLE, evidence_text STRING, curation_status STRING);",
            "CREATE REL TABLE IF NOT EXISTS IS_MARKER_FOR(FROM Entity TO Entity, log2_fold_change DOUBLE, p_value DOUBLE, cell_type STRING, confidence DOUBLE);",
        ]

        for q in schema_queries:
            self.execute_query(q)

        return schema_queries

    def clear_graph(self) -> None:
        """Clear all nodes and relationships from the database."""
        try:
            self.execute_query("MATCH (a)-[r]->(b) DELETE r;")
        except Exception:
            pass
        try:
            self.execute_query("MATCH (n) DELETE n;")
        except Exception:
            pass

    def bulk_upsert_nodes(
        self,
        label: str,
        nodes: List[Dict[str, Any]],
        id_property: Optional[str] = None,
    ) -> int:
        """Bulk upsert node records into Kùzu using UNWIND and MERGE.

        Args:
            label: Node table label ('Entity' or 'Paper').
            nodes: List of node dictionaries.
            id_property: Primary key property name (inferred if None).

        Returns:
            Count of upserted nodes.
        """
        if not nodes:
            return 0

        if id_property is None:
            if label.lower() in ("paper", "document"):
                id_property = "doi"
            elif label.lower() == "entity":
                id_property = "canonical_id"
            else:
                id_property = (
                    "canonical_id"
                    if "canonical_id" in nodes[0]
                    else ("doi" if "doi" in nodes[0] else "id")
                )

        formatted_nodes = []
        for n in nodes:
            item = dict(n)
            if "synonyms" in item and isinstance(item["synonyms"], str):
                item["synonyms"] = [item["synonyms"]]
            elif "synonyms" in item and item["synonyms"] is None:
                item["synonyms"] = []
            formatted_nodes.append(item)

        keys = [k for k in formatted_nodes[0].keys() if k != id_property]
        if keys:
            set_items = [f"n.{k} = batch.{k}" for k in keys]
            set_clause = f"SET {', '.join(set_items)}"
        else:
            set_clause = ""

        query = f"UNWIND $nodes AS batch MERGE (n:`{label}` {{{id_property}: batch.{id_property}}}) {set_clause}"
        self.execute_query(query, {"nodes": formatted_nodes})
        return len(nodes)

    def bulk_upsert_edges(
        self,
        rel_type: str,
        edges: List[Dict[str, Any]],
        source_label: str = "Entity",
        target_label: str = "Entity",
        source_key: str = "canonical_id",
        target_key: str = "canonical_id",
    ) -> int:
        """Bulk upsert relationship records into Kùzu using UNWIND and MERGE.

        Args:
            rel_type: Relationship table name ('ASSOCIATED_WITH', 'IS_MARKER_FOR').
            edges: List of relationship dictionaries.
            source_label: Label of source node table.
            target_label: Label of target node table.
            source_key: Primary key of source node.
            target_key: Primary key of target node.

        Returns:
            Count of upserted edges.
        """
        if not edges:
            return 0

        formatted_edges = []
        for e in edges:
            src_id = (
                e.get("source_id")
                or e.get("source_canonical_id")
                or e.get("source_doi")
                or e.get("source")
            )
            dst_id = (
                e.get("target_id")
                or e.get("target_canonical_id")
                or e.get("target_doi")
                or e.get("target")
            )

            if src_id is None or dst_id is None:
                continue

            item = dict(e)
            item["source_id"] = str(src_id)
            item["target_id"] = str(dst_id)
            formatted_edges.append(item)

        if not formatted_edges:
            return 0

        reserved = (
            "source_id",
            "target_id",
            "source_canonical_id",
            "target_canonical_id",
            "source_doi",
            "target_doi",
            "source",
            "target",
        )
        keys = [k for k in formatted_edges[0].keys() if k not in reserved]
        if keys:
            set_items = [f"r.{k} = batch.{k}" for k in keys]
            set_clause = f"SET {', '.join(set_items)}"
        else:
            set_clause = ""

        query = (
            f"UNWIND $edges AS batch "
            f"MATCH (a:`{source_label}` {{{source_key}: batch.source_id}}), "
            f"(b:`{target_label}` {{{target_key}: batch.target_id}}) "
            f"MERGE (a)-[r:`{rel_type}`]->(b) "
            f"{set_clause}"
        )

        self.execute_query(query, {"edges": formatted_edges})
        return len(edges)

    def bulk_upsert_relationships(
        self,
        rel_type: str,
        relationships: List[Dict[str, Any]],
        source_label: str = "Entity",
        target_label: str = "Entity",
        source_key: str = "canonical_id",
        target_key: str = "canonical_id",
    ) -> int:
        """Alias for bulk_upsert_edges."""
        return self.bulk_upsert_edges(
            rel_type=rel_type,
            edges=relationships,
            source_label=source_label,
            target_label=target_label,
            source_key=source_key,
            target_key=target_key,
        )

    def get_2hop_neighborhood(self, entity_id: str) -> Dict[str, Any]:
        """Retrieve the 2-hop neighborhood path structure for a given entity_id.

        Args:
            entity_id: Canonical ID of the start entity.

        Returns:
            Dictionary containing start entity, 1-hop nodes, 2-hop nodes, relationships, and path chains.
        """
        cypher = """
        MATCH (start:Entity {canonical_id: $entity_id})-[r1]-(hop1:Entity)
        OPTIONAL MATCH (hop1)-[r2]-(hop2:Entity)
        WHERE hop2.canonical_id <> start.canonical_id
        RETURN start.canonical_id AS start_id,
               label(r1) AS rel1_type,
               hop1.canonical_id AS hop1_id,
               hop1.name AS hop1_name,
               hop1.entity_type AS hop1_type,
               label(r2) AS rel2_type,
               hop2.canonical_id AS hop2_id,
               hop2.name AS hop2_name,
               hop2.entity_type AS hop2_type
        """
        rows = self.execute_query(cypher, {"entity_id": entity_id})

        hop1_nodes: Dict[str, Dict[str, Any]] = {}
        hop2_nodes: Dict[str, Dict[str, Any]] = {}
        relationships: List[Dict[str, Any]] = []
        paths: List[Dict[str, Any]] = []

        for r in rows:
            h1_id = r.get("hop1_id")
            if h1_id:
                hop1_nodes[h1_id] = {
                    "canonical_id": h1_id,
                    "name": r.get("hop1_name"),
                    "entity_type": r.get("hop1_type"),
                }
                relationships.append(
                    {
                        "source": entity_id,
                        "target": h1_id,
                        "type": r.get("rel1_type"),
                    }
                )

            h2_id = r.get("hop2_id")
            if h2_id and h2_id != entity_id and h2_id not in hop1_nodes:
                hop2_nodes[h2_id] = {
                    "canonical_id": h2_id,
                    "name": r.get("hop2_name"),
                    "entity_type": r.get("hop2_type"),
                }
                if h1_id:
                    relationships.append(
                        {
                            "source": h1_id,
                            "target": h2_id,
                            "type": r.get("rel2_type"),
                        }
                    )

            paths.append(
                {
                    "start": entity_id,
                    "hop1": h1_id,
                    "hop2": h2_id,
                    "rel1": r.get("rel1_type"),
                    "rel2": r.get("rel2_type"),
                }
            )

        return {
            "start_id": entity_id,
            "hop1_nodes": list(hop1_nodes.values()),
            "hop2_nodes": list(hop2_nodes.values()),
            "relationships": relationships,
            "paths": paths,
        }

    def get_neighborhood_paths(
        self, entity_ids: Union[str, List[str]], max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """Retrieve 2-hop neighborhood paths for one or more entity IDs.

        Args:
            entity_ids: Single entity ID or list of entity IDs.
            max_hops: Maximum traversal hops (default 2).

        Returns:
            List of path dictionaries.
        """
        if isinstance(entity_ids, str):
            ids = [entity_ids]
        else:
            ids = list(entity_ids)

        all_paths: List[Dict[str, Any]] = []
        for eid in ids:
            neighborhood = self.get_2hop_neighborhood(eid)
            all_paths.extend(neighborhood.get("paths", []))

        return all_paths
