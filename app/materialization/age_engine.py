"""Apache AGE PostgreSQL Graph Database Engine Wrapper for EBV Knowledge System.

Provides graph database execution over PostgreSQL Apache AGE extension with Cypher query support,
schema management, bulk write operations, and 2-hop neighborhood path retrieval.
Includes robust fallback mock execution when Apache AGE or PostgreSQL is unavailable.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Attempt to import psycopg safely
try:
    import psycopg
    PSYCOPG_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    psycopg = None
    PSYCOPG_AVAILABLE = False


class MockAgeQueryResult:
    """Mock QueryResult for Apache AGE fallback mode."""

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

    def fetchone(self) -> Optional[Dict[str, Any]]:
        if not self.has_next():
            return None
        row = self._rows[self._index]
        self._index += 1
        return row

    def get_next(self) -> List[Any]:
        if not self.has_next():
            raise StopIteration("No more rows in MockAgeQueryResult")
        row = self._rows[self._index]
        self._index += 1
        return [row.get(col) for col in self._column_names]

    def fetchall(self) -> List[Dict[str, Any]]:
        return self._rows

    def rows_as_dict(self) -> List[Dict[str, Any]]:
        return self._rows


class MockAgeDatabase:
    """Mock database store for Apache AGE fallback mode."""

    def __init__(self, graph_name: str = "ebv_kg") -> None:
        self.graph_name = graph_name
        self.is_closed = False
        self.nodes: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.relationships: Dict[str, List[Dict[str, Any]]] = {}

    def close(self) -> None:
        self.is_closed = True


class MockAgeConnection:
    """Mock Connection class for Apache AGE fallback mode."""

    def __init__(self, db: MockAgeDatabase) -> None:
        self.db = db
        self.is_closed = False

    def close(self) -> None:
        self.is_closed = True

    def execute(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> MockAgeQueryResult:
        parameters = parameters or {}
        q_upper = query.upper()

        if "CREATE" in q_upper and ("GRAPH" in q_upper or "LABEL" in q_upper or "CONSTRAINT" in q_upper):
            return MockAgeQueryResult([])

        if "UNWIND" in q_upper or "MERGE" in q_upper or "CREATE" in q_upper:
            if "nodes" in parameters:
                node_list = parameters["nodes"]
                label = "Entity"
                for l in ["Entity", "Paper"]:
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
                return MockAgeQueryResult([])

            elif "edges" in parameters or "relationships" in parameters:
                edge_list = (
                    parameters.get("edges") or parameters.get("relationships") or []
                )
                rel_type = "ASSOCIATED_WITH"
                for r in ["ASSOCIATED_WITH", "IS_MARKER_FOR", "TARGETS", "INTERACTS_WITH"]:
                    if f":{r}" in query or f":`{r}`" in query:
                        rel_type = r
                        break
                if rel_type not in self.db.relationships:
                    self.db.relationships[rel_type] = []

                for edge in edge_list:
                    self.db.relationships[rel_type].append(edge)
                return MockAgeQueryResult([])

        if "DETACH DELETE" in q_upper or "DELETE" in q_upper:
            self.db.nodes.clear()
            self.db.relationships.clear()
            return MockAgeQueryResult([])

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
                                }
                            )
                return MockAgeQueryResult(rows)

            rows = []
            matching_labels = [l for l in self.db.nodes.keys() if f":{l}" in query or f":`{l}`" in query]
            if not matching_labels:
                matching_labels = list(self.db.nodes.keys())

            for label in matching_labels:
                node_dict = self.db.nodes.get(label, {})
                for node_id, node_data in node_dict.items():
                    if (
                        entity_id is None
                        or node_id == entity_id
                        or node_data.get("name") == entity_id
                    ):
                        row = dict(node_data)
                        row["id"] = (
                            node_data.get("canonical_id")
                            or node_data.get("doi")
                            or node_data.get("id")
                            or node_id
                        )
                        rows.append(row)
            return MockAgeQueryResult(rows)

        return MockAgeQueryResult([])


class AgeEngine:
    """Wrapper around PostgreSQL Apache AGE graph database engine."""

    def __init__(
        self,
        dsn: Optional[str] = None,
        graph_name: str = "ebv_kg",
        connection: Any = None,
        force_mock: bool = False,
    ) -> None:
        """Initialize AgeEngine.

        Args:
            dsn: PostgreSQL connection DSN string (e.g. postgresql://user:pass@localhost:5432/dbname).
            graph_name: Name of Apache AGE graph.
            connection: Existing psycopg connection object if provided.
            force_mock: If True, forces use of Mock AGE engine.
        """
        self.dsn = dsn or os.getenv("POSTGRES_DSN", "postgresql://postgres:postgres@localhost:5432/ebv_kg")
        self.graph_name = graph_name
        self.is_mock = force_mock

        if not self.is_mock and connection is not None:
            self.conn = connection
        elif not self.is_mock and PSYCOPG_AVAILABLE:
            try:
                logger.info("Connecting to PostgreSQL Apache AGE at %s", self.dsn)
                self.conn = psycopg.connect(self.dsn)
            except Exception as e:
                logger.warning("Could not connect to PostgreSQL AGE (%s). Falling back to Mock AGE.", e)
                self.is_mock = True

        if self.is_mock:
            logger.info("Initializing fallback Mock Apache AGE engine")
            self.db = MockAgeDatabase(graph_name=graph_name)
            self.conn = MockAgeConnection(self.db)

    def close(self) -> None:
        """Close the underlying database connection."""
        if hasattr(self, "conn") and self.conn is not None:
            try:
                self.conn.close()
            except Exception as e:
                logger.warning("Error closing AgeEngine connection: %s", e)

    def __enter__(self) -> "AgeEngine":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def execute_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query on Apache AGE or mock store and return dictionary results."""
        if self.conn is None:
            raise RuntimeError("AgeEngine connection is not initialized.")
        parameters = parameters or {}

        if self.is_mock:
            result = self.conn.execute(query, parameters)
            return result.rows_as_dict()

        try:
            with self.conn.cursor() as cur:
                cur.execute("LOAD 'age';")
                cur.execute("SET search_path = ag_catalog, '$user', public;")
                sql = f"SELECT * FROM cypher('{self.graph_name}', %s) AS (result agtype);"
                cur.execute(sql, (query,))
                rows = cur.fetchall()
                return [{"result": r[0]} for r in rows]
        except Exception as e:
            logger.error("Error executing Cypher query in Apache AGE: %s", e)
            raise

    def execute_cypher(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Alias for execute_query."""
        return self.execute_query(query, parameters)

    def init_schema(self) -> List[str]:
        """Initialize Apache AGE graph schema.

        Returns:
            List of executed DDL / schema setup query strings.
        """
        schema_queries = [
            f"SELECT create_graph('{self.graph_name}');",
            f"SELECT create_vlabel('{self.graph_name}', 'Entity');",
            f"SELECT create_vlabel('{self.graph_name}', 'Paper');",
            f"SELECT create_elabel('{self.graph_name}', 'ASSOCIATED_WITH');",
            f"SELECT create_elabel('{self.graph_name}', 'IS_MARKER_FOR');",
        ]

        if self.is_mock:
            for q in schema_queries:
                self.execute_query(q)
            return schema_queries

        try:
            with self.conn.cursor() as cur:
                cur.execute("LOAD 'age';")
                cur.execute("SET search_path = ag_catalog, '$user', public;")
                cur.execute(
                    "SELECT count(*) FROM ag_graph WHERE name = %s;", (self.graph_name,)
                )
                exists = cur.fetchone()[0] > 0
                if not exists:
                    cur.execute(f"SELECT create_graph('{self.graph_name}');")
            self.conn.commit()
        except Exception as e:
            logger.warning("AGE schema initialization warning: %s", e)

        return schema_queries

    def clear_graph(self) -> None:
        """Clear all nodes and relationships from the database."""
        if self.is_mock:
            self.execute_query("MATCH (n) DETACH DELETE n;")
            return

        try:
            with self.conn.cursor() as cur:
                cur.execute("LOAD 'age';")
                cur.execute("SET search_path = ag_catalog, '$user', public;")
                cur.execute(f"SELECT drop_graph('{self.graph_name}', true);")
                cur.execute(f"SELECT create_graph('{self.graph_name}');")
            self.conn.commit()
        except Exception as e:
            logger.warning("Error clearing Apache AGE graph: %s", e)

    def bulk_upsert_nodes(
        self,
        label: str,
        nodes: List[Dict[str, Any]],
        id_property: Optional[str] = None,
    ) -> int:
        """Bulk upsert node records into Apache AGE.

        Args:
            label: Node label ('Entity' or 'Paper').
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

        query = f"UNWIND $nodes AS batch MERGE (n:`{label}` {{{id_property}: batch.{id_property}}}) SET n += batch"
        self.execute_query(query, {"nodes": nodes})
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
        """Bulk upsert relationship records into Apache AGE.

        Args:
            rel_type: Relationship type ('ASSOCIATED_WITH', 'IS_MARKER_FOR').
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

        query = (
            f"UNWIND $edges AS batch "
            f"MATCH (a:`{source_label}` {{{source_key}: batch.source_id}}), "
            f"(b:`{target_label}` {{{target_key}: batch.target_id}}) "
            f"MERGE (a)-[r:`{rel_type}`]->(b) "
            f"SET r += batch"
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
        """Retrieve the 2-hop neighborhood path structure for a given entity_id."""
        cypher = """
        MATCH (start:Entity {canonical_id: $entity_id})-[r1]-(hop1:Entity)
        OPTIONAL MATCH (hop1)-[r2]-(hop2:Entity)
        WHERE hop2.canonical_id <> start.canonical_id
        RETURN start.canonical_id AS start_id,
               label(r1) AS rel1_type,
               hop1.canonical_id AS hop1_id,
               hop1.name AS hop1_name,
               label(r2) AS rel2_type,
               hop2.canonical_id AS hop2_id,
               hop2.name AS hop2_name
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
        """Retrieve 2-hop neighborhood paths for one or more entity IDs."""
        if isinstance(entity_ids, str):
            ids = [entity_ids]
        else:
            ids = list(entity_ids)

        all_paths: List[Dict[str, Any]] = []
        for eid in ids:
            neighborhood = self.get_2hop_neighborhood(eid)
            all_paths.extend(neighborhood.get("paths", []))

        return all_paths
