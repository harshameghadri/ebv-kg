import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Set
from app.materialization.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

class DiscoveryHarvester:
    def __init__(self, pg_conn: Any, neo4j_client: Optional[Neo4jClient] = None) -> None:
        self.pg_conn = pg_conn
        self.neo4j_client = neo4j_client

    def harvest_candidates(
        self,
        communities: List[List[Dict[str, Any]]],
        export_json_path: Optional[str] = None,
        insert_to_db: bool = True,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        # Fetch Neo4j relationships
        neo4j_edges: Set[tuple] = set()
        if self.neo4j_client:
            try:
                cypher = "MATCH (a:Entity)-[r]->(b:Entity) RETURN a.canonical_id AS source, b.canonical_id AS target"
                results = self.neo4j_client.execute_query(cypher)
                for record in results:
                    try:
                        source = record.get("source")
                        target = record.get("target")
                    except AttributeError:
                        rec_dict = dict(record)
                        source = rec_dict.get("source")
                        target = rec_dict.get("target")
                    if source and target:
                        neo4j_edges.add((source, target))
                        neo4j_edges.add((target, source))
            except Exception as e:
                logger.warning(f"Error fetching relationships from Neo4j: {e}")

        # Fetch Postgres entities & relationships
        allowed_types = {"GENE", "DISEASE", "CHEMICAL", "CELL_TYPE"}
        entities = {}
        with self.pg_conn.cursor() as cur:
            cur.execute("SELECT id, canonical_id, name, entity_type FROM normalized_entities")
            for row in cur.fetchall():
                ent_id, canonical_id, name, entity_type = row
                if entity_type and entity_type.upper() in allowed_types:
                    entities[ent_id] = {
                        "id": ent_id,
                        "canonical_id": canonical_id,
                        "name": name,
                        "entity_type": entity_type.upper()
                    }

        adj = {str(ent_id): set() for ent_id in entities}
        existing_pg_relationships = set()
        with self.pg_conn.cursor() as cur:
            cur.execute("SELECT source_entity_id, target_entity_id FROM relationships")
            for row in cur.fetchall():
                src_id, tgt_id = str(row[0]), str(row[1])
                if src_id in adj and tgt_id in adj:
                    adj[src_id].add(tgt_id)
                    adj[tgt_id].add(src_id)
                existing_pg_relationships.add((src_id, tgt_id))
                existing_pg_relationships.add((tgt_id, src_id))

        candidates = []
        seen_pairs = set()

        for community in communities:
            n = len(community)
            for i in range(n):
                for j in range(i + 1, n):
                    ent1 = community[i]
                    ent2 = community[j]

                    id1_str = str(ent1["id"])
                    id2_str = str(ent2["id"])

                    pair = tuple(sorted([id1_str, id2_str]))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)

                    if (id1_str, id2_str) in existing_pg_relationships or (id2_str, id1_str) in existing_pg_relationships:
                        continue

                    cid1 = ent1["canonical_id"]
                    cid2 = ent2["canonical_id"]
                    if (cid1, cid2) in neo4j_edges or (cid2, cid1) in neo4j_edges:
                        continue

                    n1 = adj.get(id1_str, set())
                    n2 = adj.get(id2_str, set())

                    shared = n1 & n2
                    union = n1 | n2

                    shared_count = len(shared)
                    jaccard = shared_count / len(union) if union else 0.0
                    degree_sum = len(n1) + len(n2)

                    discovery_score = shared_count + 0.5 * jaccard + 0.01 * degree_sum

                    candidates.append({
                        "source_entity_id": ent1["id"],
                        "source_entity": ent1,
                        "target_entity_id": ent2["id"],
                        "target_entity": ent2,
                        "discovery_score": discovery_score,
                        "relationship_type": "ASSOCIATED_WITH"
                    })

        candidates.sort(key=lambda x: x["discovery_score"], reverse=True)
        top_candidates = candidates[:limit]

        if insert_to_db and top_candidates:
            with self.pg_conn.cursor() as cur:
                for cand in top_candidates:
                    rel_id = uuid.uuid4()
                    cur.execute(
                        "INSERT INTO relationships (id, source_entity_id, target_entity_id, relationship_type, confidence_score, curation_status, source_type) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        (
                            rel_id,
                            cand["source_entity_id"],
                            cand["target_entity_id"],
                            cand["relationship_type"],
                            cand["discovery_score"],
                            "PENDING",
                            "automated_discovery"
                        )
                    )
            if hasattr(self.pg_conn, "commit"):
                self.pg_conn.commit()

        if export_json_path:
            serializable = []
            for cand in top_candidates:
                serializable.append({
                    "source_entity_id": str(cand["source_entity_id"]),
                    "source_entity": {
                        "id": str(cand["source_entity"]["id"]),
                        "canonical_id": cand["source_entity"]["canonical_id"],
                        "name": cand["source_entity"]["name"],
                        "entity_type": cand["source_entity"]["entity_type"]
                    },
                    "target_entity_id": str(cand["target_entity_id"]),
                    "target_entity": {
                        "id": str(cand["target_entity"]["id"]),
                        "canonical_id": cand["target_entity"]["canonical_id"],
                        "name": cand["target_entity"]["name"],
                        "entity_type": cand["target_entity"]["entity_type"]
                    },
                    "discovery_score": cand["discovery_score"],
                    "relationship_type": cand["relationship_type"]
                })
            with open(export_json_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)

        return top_candidates
