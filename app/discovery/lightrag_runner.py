import logging
import random
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

class DiscoveryClusteringRunner:
    def __init__(self, pg_conn: Any) -> None:
        self.pg_conn = pg_conn

    def _fetch_biological_entities(self) -> Dict[Any, Dict[str, Any]]:
        allowed_types = {"GENE", "DISEASE", "CHEMICAL", "CELL_TYPE"}
        entities_dict = {}
        with self.pg_conn.cursor() as cur:
            cur.execute(
                "SELECT id, canonical_id, name, entity_type FROM normalized_entities"
            )
            rows = cur.fetchall()
            for row in rows:
                ent_id, canonical_id, name, entity_type = row
                if entity_type and entity_type.upper() in allowed_types:
                    entities_dict[ent_id] = {
                        "id": ent_id,
                        "canonical_id": canonical_id,
                        "name": name,
                        "entity_type": entity_type.upper(),
                    }
        return entities_dict

    def _fetch_relationships(self, entities_dict: Dict[Any, Dict[str, Any]], min_confidence: float) -> List[Tuple[Any, Any, float]]:
        edges = []
        with self.pg_conn.cursor() as cur:
            cur.execute(
                "SELECT source_entity_id, target_entity_id, confidence_score FROM relationships"
            )
            rows = cur.fetchall()
            for row in rows:
                src_id, tgt_id, conf = row
                if src_id in entities_dict and tgt_id in entities_dict:
                    confidence = conf if conf is not None else 1.0
                    if confidence >= min_confidence:
                        edges.append((src_id, tgt_id, confidence))
        return edges

    def run_clustering(
        self,
        min_confidence: float = 0.0,
        algorithm: str = "label_propagation",
        max_iter: int = 20,
    ) -> List[List[Dict[str, Any]]]:
        entities_dict = self._fetch_biological_entities()
        if not entities_dict:
            logger.warning("No normalized biological entities found in database.")
            return []

        edges = self._fetch_relationships(entities_dict, min_confidence)
        nodes = list(entities_dict.keys())

        if algorithm == "connected_components":
            communities_keys = self._cluster_connected_components(nodes, edges)
        elif algorithm == "label_propagation":
            communities_keys = self._cluster_label_propagation(nodes, edges, max_iter)
        else:
            raise ValueError(f"Unknown clustering algorithm: {algorithm}")

        communities = []
        for comm_keys in communities_keys:
            if not comm_keys:
                continue
            comm_list = [entities_dict[k] for k in comm_keys if k in entities_dict]
            if comm_list:
                comm_list.sort(key=lambda x: x["canonical_id"])
                communities.append(comm_list)

        communities.sort(key=lambda x: (-len(x), x[0]["canonical_id"]))
        return communities

    def _cluster_connected_components(
        self, nodes: List[Any], edges: List[Tuple[Any, Any, float]]
    ) -> List[Set[Any]]:
        adj: Dict[Any, Set[Any]] = {node: set() for node in nodes}
        for u, v, _ in edges:
            if u in adj and v in adj:
                adj[u].add(v)
                adj[v].add(u)

        visited: Set[Any] = set()
        components: List[Set[Any]] = []

        for node in nodes:
            if node not in visited:
                component = set()
                queue = [node]
                visited.add(node)
                while queue:
                    curr = queue.pop(0)
                    component.add(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                components.append(component)

        return components

    def _cluster_label_propagation(
        self, nodes: List[Any], edges: List[Tuple[Any, Any, float]], max_iter: int
    ) -> List[Set[Any]]:
        labels = {node: node for node in nodes}
        adj: Dict[Any, Dict[Any, float]] = {node: {} for node in nodes}
        for u, v, w in edges:
            if u in adj and v in adj:
                adj[u][v] = w
                adj[v][u] = w

        rng = random.Random(42)

        for _ in range(max_iter):
            shuffled_nodes = list(nodes)
            rng.shuffle(shuffled_nodes)

            changed = False
            for node in shuffled_nodes:
                if not adj[node]:
                    continue

                label_weights: Dict[Any, float] = {}
                for neighbor, weight in adj[node].items():
                    nl = labels[neighbor]
                    label_weights[nl] = label_weights.get(nl, 0.0) + weight

                max_weight = -1.0
                best_labels: List[Any] = []
                for label, w in label_weights.items():
                    if w > max_weight:
                        max_weight = w
                        best_labels = [label]
                    elif w == max_weight:
                        best_labels.append(label)

                if best_labels:
                    new_label = rng.choice(best_labels)
                    if labels[node] != new_label:
                        labels[node] = new_label
                        changed = True

            if not changed:
                break

        communities_map: Dict[Any, Set[Any]] = {}
        for node, label in labels.items():
            communities_map.setdefault(label, set()).add(node)

        return list(communities_map.values())
