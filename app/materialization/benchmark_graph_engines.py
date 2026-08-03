"""Graph Engine Benchmarking Module for EBV Knowledge System.

Measures write throughput (nodes/sec, edges/sec) and multi-hop Cypher read latency (1-hop, 2-hop)
across Neo4jClient, KuzuEngine, and AgeEngine graph database engines.
Outputs a structured JSON benchmark metrics report.
"""

import argparse
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.materialization.age_engine import AgeEngine
from app.materialization.kuzu_engine import KuzuEngine
from app.materialization.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


def generate_benchmark_dataset(
    num_nodes: int = 100, num_edges: int = 200
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate synthetic nodes and edges for benchmarking realistic graph workload.

    Args:
        num_nodes: Number of Entity nodes to generate.
        num_edges: Number of ASSOCIATED_WITH relationships to generate.

    Returns:
        Tuple of (nodes_list, edges_list).
    """
    nodes = []
    entity_types = ["Gene", "DiseaseOutcome", "CellState", "ViralProtein"]
    for i in range(num_nodes):
        nodes.append(
            {
                "canonical_id": f"ENT_{i:05d}",
                "name": f"Entity_{i}",
                "entity_type": entity_types[i % len(entity_types)],
                "ontology_source": "HGNC" if i % 2 == 0 else "DOID",
                "synonyms": [f"syn_{i}_1", f"syn_{i}_2"],
            }
        )

    edges = []
    if num_nodes > 1:
        for j in range(num_edges):
            src_idx = j % num_nodes
            dst_idx = (j * 7 + 1) % num_nodes
            if src_idx == dst_idx:
                dst_idx = (src_idx + 1) % num_nodes

            edges.append(
                {
                    "source_canonical_id": f"ENT_{src_idx:05d}",
                    "target_canonical_id": f"ENT_{dst_idx:05d}",
                    "source_id": f"ENT_{src_idx:05d}",
                    "target_id": f"ENT_{dst_idx:05d}",
                    "relationship_type": "ASSOCIATED_WITH",
                    "confidence": round(0.5 + (j % 50) / 100.0, 2),
                    "evidence_text": f"Benchmark relationship {j}",
                    "curation_status": "APPROVED",
                }
            )

    return nodes, edges


def calculate_percentiles(latencies_ms: List[float]) -> Dict[str, float]:
    """Calculate mean, median, p95, p99, min, and max from a list of latency measurements.

    Args:
        latencies_ms: List of latency values in milliseconds.

    Returns:
        Dictionary of summary statistics.
    """
    if not latencies_ms:
        return {
            "mean_latency_ms": 0.0,
            "median_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "min_latency_ms": 0.0,
            "max_latency_ms": 0.0,
        }

    sorted_lats = sorted(latencies_ms)
    n = len(sorted_lats)

    def percentile(p: float) -> float:
        k = (n - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_lats[int(k)]
        return sorted_lats[int(f)] * (c - k) + sorted_lats[int(c)] * (k - f)

    mean_val = sum(sorted_lats) / n
    median_val = percentile(50.0)
    p95_val = percentile(95.0)
    p99_val = percentile(99.0)

    return {
        "mean_latency_ms": round(mean_val, 4),
        "median_latency_ms": round(median_val, 4),
        "p95_latency_ms": round(p95_val, 4),
        "p99_latency_ms": round(p99_val, 4),
        "min_latency_ms": round(sorted_lats[0], 4),
        "max_latency_ms": round(sorted_lats[-1], 4),
    }


class GraphEngineBenchmark:
    """Benchmarking runner for measuring graph engine performance."""

    def __init__(self, force_mock: bool = False) -> None:
        self.force_mock = force_mock

    def benchmark_engine(
        self,
        engine_name: str,
        engine_instance: Any,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        num_read_iterations: int = 10,
    ) -> Dict[str, Any]:
        """Run write throughput and multi-hop Cypher read latency benchmark on a single graph engine.

        Args:
            engine_name: Name of engine ('Neo4jClient', 'KuzuEngine', or 'AgeEngine').
            engine_instance: Graph engine client instance.
            nodes: List of node dicts to write.
            edges: List of edge dicts to write.
            num_read_iterations: Number of read query repetitions.

        Returns:
            Dictionary containing structured benchmark metrics for this engine.
        """
        logger.info("Benchmarking graph engine: %s", engine_name)

        # 1. Initialize schema & clear graph
        try:
            engine_instance.init_schema()
        except Exception as e:
            logger.warning("Error initializing schema on %s: %s", engine_name, e)

        try:
            engine_instance.clear_graph()
        except Exception as e:
            logger.warning("Error clearing graph on %s: %s", engine_name, e)

        # 2. Node write throughput
        t0 = time.perf_counter()
        nodes_upserted = engine_instance.bulk_upsert_nodes("Entity", nodes)
        t1 = time.perf_counter()
        node_write_time = max(t1 - t0, 1e-6)
        nodes_count = nodes_upserted if isinstance(nodes_upserted, int) and nodes_upserted > 0 else len(nodes)
        nodes_per_sec = nodes_count / node_write_time

        # 3. Edge write throughput
        t0 = time.perf_counter()
        edges_upserted = engine_instance.bulk_upsert_edges("ASSOCIATED_WITH", edges)
        t1 = time.perf_counter()
        edge_write_time = max(t1 - t0, 1e-6)
        edges_count = edges_upserted if isinstance(edges_upserted, int) and edges_upserted > 0 else len(edges)
        edges_per_sec = edges_count / edge_write_time

        # Sample query node IDs
        sample_ids = [n["canonical_id"] for n in nodes[:min(10, len(nodes))]] if nodes else ["ENT_00000"]

        # 4. Multi-hop 1-hop Cypher read latency
        cypher_1hop = (
            "MATCH (n:Entity {canonical_id: $entity_id})-[r:ASSOCIATED_WITH]-(m:Entity) "
            "RETURN n.canonical_id AS start_id, m.canonical_id AS hop1_id"
        )
        lats_1hop: List[float] = []

        for i in range(num_read_iterations):
            entity_id = sample_ids[i % len(sample_ids)]
            t0 = time.perf_counter()
            try:
                if hasattr(engine_instance, "execute_query"):
                    engine_instance.execute_query(cypher_1hop, {"entity_id": entity_id})
                elif hasattr(engine_instance, "get_2hop_neighborhood"):
                    engine_instance.get_2hop_neighborhood(entity_id)
            except Exception as e:
                logger.warning("1-hop query execution fallback on %s: %s", engine_name, e)
            t1 = time.perf_counter()
            lats_1hop.append((t1 - t0) * 1000.0)

        # 5. Multi-hop 2-hop Cypher read latency
        cypher_2hop = (
            "MATCH (n:Entity {canonical_id: $entity_id})-[r1:ASSOCIATED_WITH]-(m:Entity)-[r2:ASSOCIATED_WITH]-(k:Entity) "
            "WHERE k.canonical_id <> n.canonical_id "
            "RETURN n.canonical_id AS start_id, m.canonical_id AS hop1_id, k.canonical_id AS hop2_id"
        )
        lats_2hop: List[float] = []

        for i in range(num_read_iterations):
            entity_id = sample_ids[i % len(sample_ids)]
            t0 = time.perf_counter()
            try:
                if hasattr(engine_instance, "get_2hop_neighborhood"):
                    engine_instance.get_2hop_neighborhood(entity_id)
                elif hasattr(engine_instance, "execute_query"):
                    engine_instance.execute_query(cypher_2hop, {"entity_id": entity_id})
            except Exception as e:
                logger.warning("2-hop query execution fallback on %s: %s", engine_name, e)
            t1 = time.perf_counter()
            lats_2hop.append((t1 - t0) * 1000.0)

        metrics_1hop = calculate_percentiles(lats_1hop)
        metrics_1hop["iterations"] = num_read_iterations

        metrics_2hop = calculate_percentiles(lats_2hop)
        metrics_2hop["iterations"] = num_read_iterations

        is_mock = getattr(engine_instance, "is_mock", False)

        return {
            "engine": engine_name,
            "is_mock": is_mock,
            "node_write": {
                "count": nodes_count,
                "total_time_sec": round(node_write_time, 6),
                "throughput_nodes_per_sec": round(nodes_per_sec, 2),
            },
            "edge_write": {
                "count": edges_count,
                "total_time_sec": round(edge_write_time, 6),
                "throughput_edges_per_sec": round(edges_per_sec, 2),
            },
            "read_1hop": metrics_1hop,
            "read_2hop": metrics_2hop,
        }


def mock_neo4j_client_instance() -> Neo4jClient:
    """Create a Neo4jClient instance backed by a mock driver."""
    from unittest.mock import MagicMock
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    session.run.return_value = []
    client = Neo4jClient(driver=driver)
    client.is_mock = True
    return client


def run_graph_engine_benchmark(
    engines: Optional[List[str]] = None,
    num_nodes: int = 100,
    num_edges: int = 200,
    num_read_iterations: int = 10,
    output_path: Optional[str] = None,
    force_mock: bool = True,
) -> Dict[str, Any]:
    """Execute benchmarking suite across Neo4jClient, KuzuEngine, and AgeEngine.

    Args:
        engines: List of engine names to benchmark (default: ['Neo4jClient', 'KuzuEngine', 'AgeEngine']).
        num_nodes: Count of nodes to test.
        num_edges: Count of edges to test.
        num_read_iterations: Number of read query repetitions per test.
        output_path: Optional path to save structured JSON report file.
        force_mock: If True, uses mock execution mode for all engines.

    Returns:
        Structured JSON benchmark report dictionary.
    """
    target_engines = engines or ["Neo4jClient", "KuzuEngine", "AgeEngine"]
    nodes, edges = generate_benchmark_dataset(num_nodes=num_nodes, num_edges=num_edges)

    runner = GraphEngineBenchmark(force_mock=force_mock)
    engine_results: Dict[str, Any] = {}

    for name in target_engines:
        engine_inst = None
        try:
            if name == "Neo4jClient":
                if force_mock:
                    engine_inst = mock_neo4j_client_instance()
                else:
                    try:
                        engine_inst = Neo4jClient()
                        engine_inst.is_mock = False
                    except Exception:
                        engine_inst = mock_neo4j_client_instance()

            elif name == "KuzuEngine":
                engine_inst = KuzuEngine(db_path=":memory:", force_mock=force_mock)

            elif name == "AgeEngine":
                engine_inst = AgeEngine(force_mock=force_mock)

            else:
                logger.warning("Unknown engine name '%s', skipping", name)
                continue

            res = runner.benchmark_engine(
                engine_name=name,
                engine_instance=engine_inst,
                nodes=nodes,
                edges=edges,
                num_read_iterations=num_read_iterations,
            )
            engine_results[name] = res

        finally:
            if engine_inst and hasattr(engine_inst, "close"):
                try:
                    engine_inst.close()
                except Exception:
                    pass

    fastest_node_write = max(
        engine_results.keys(),
        key=lambda k: engine_results[k]["node_write"]["throughput_nodes_per_sec"],
        default=None,
    ) if engine_results else None

    fastest_edge_write = max(
        engine_results.keys(),
        key=lambda k: engine_results[k]["edge_write"]["throughput_edges_per_sec"],
        default=None,
    ) if engine_results else None

    lowest_1hop = min(
        engine_results.keys(),
        key=lambda k: engine_results[k]["read_1hop"]["mean_latency_ms"],
        default=None,
    ) if engine_results else None

    lowest_2hop = min(
        engine_results.keys(),
        key=lambda k: engine_results[k]["read_2hop"]["mean_latency_ms"],
        default=None,
    ) if engine_results else None

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_read_iterations": num_read_iterations,
            "force_mock": force_mock,
        },
        "engines": engine_results,
        "summary": {
            "fastest_node_write": fastest_node_write,
            "fastest_edge_write": fastest_edge_write,
            "lowest_1hop_latency": lowest_1hop,
            "lowest_2hop_latency": lowest_2hop,
        },
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Saved benchmark report to %s", output_path)

    return report


def main() -> None:
    """CLI entry point for running graph engine benchmark."""
    parser = argparse.ArgumentParser(description="EBV Knowledge Graph Engine Benchmarking Tool")
    parser.add_argument("--nodes", type=int, default=100, help="Number of nodes to benchmark (default 100)")
    parser.add_argument("--edges", type=int, default=200, help="Number of edges to benchmark (default 200)")
    parser.add_argument("--iterations", type=int, default=10, help="Number of read query iterations (default 10)")
    parser.add_argument("--output", type=str, default=None, help="JSON output file path for benchmark report")
    parser.add_argument("--force-mock", action="store_true", default=True, help="Force mock engine execution")
    parser.add_argument(
        "--engines",
        nargs="+",
        default=["Neo4jClient", "KuzuEngine", "AgeEngine"],
        help="List of engines to benchmark",
    )

    args = parser.parse_args()
    report = run_graph_engine_benchmark(
        engines=args.engines,
        num_nodes=args.nodes,
        num_edges=args.edges,
        num_read_iterations=args.iterations,
        output_path=args.output,
        force_mock=args.force_mock,
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
