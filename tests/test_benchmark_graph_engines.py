"""Unit tests for Graph Engine Benchmarking module and AgeEngine."""

import json
import pytest

from app.materialization.age_engine import AgeEngine
from app.materialization.kuzu_engine import KuzuEngine
from app.materialization.neo4j_client import Neo4jClient
from app.materialization.benchmark_graph_engines import (
    generate_benchmark_dataset,
    calculate_percentiles,
    GraphEngineBenchmark,
    run_graph_engine_benchmark,
)


def test_generate_benchmark_dataset():
    """Verify synthetic dataset generation for nodes and edges."""
    nodes, edges = generate_benchmark_dataset(num_nodes=50, num_edges=100)
    assert len(nodes) == 50
    assert len(edges) == 100

    first_node = nodes[0]
    assert "canonical_id" in first_node
    assert "name" in first_node
    assert "entity_type" in first_node

    first_edge = edges[0]
    assert "source_canonical_id" in first_edge
    assert "target_canonical_id" in first_edge
    assert first_edge["relationship_type"] == "ASSOCIATED_WITH"


def test_calculate_percentiles():
    """Verify latency summary statistics calculations."""
    latencies = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    stats = calculate_percentiles(latencies)

    assert stats["mean_latency_ms"] == 5.5
    assert stats["median_latency_ms"] == 5.5
    assert stats["min_latency_ms"] == 1.0
    assert stats["max_latency_ms"] == 10.0
    assert stats["p95_latency_ms"] > 9.0
    assert stats["p99_latency_ms"] > 9.0


def test_calculate_percentiles_empty():
    """Verify latency calculations on empty list."""
    stats = calculate_percentiles([])
    assert stats["mean_latency_ms"] == 0.0
    assert stats["median_latency_ms"] == 0.0


def test_age_engine_mock():
    """Verify AgeEngine operations in mock mode."""
    with AgeEngine(force_mock=True) as engine:
        assert engine.is_mock is True
        schema = engine.init_schema()
        assert len(schema) > 0

        nodes = [
            {"canonical_id": "E1", "name": "Gene1", "entity_type": "Gene"},
            {"canonical_id": "E2", "name": "Disease1", "entity_type": "DiseaseOutcome"},
        ]
        n_count = engine.bulk_upsert_nodes("Entity", nodes)
        assert n_count == 2

        edges = [
            {
                "source_canonical_id": "E1",
                "target_canonical_id": "E2",
                "relationship_type": "ASSOCIATED_WITH",
            }
        ]
        e_count = engine.bulk_upsert_edges("ASSOCIATED_WITH", edges)
        assert e_count == 1

        neighborhood = engine.get_2hop_neighborhood("E1")
        assert neighborhood["start_id"] == "E1"
        assert len(neighborhood["hop1_nodes"]) == 1

        engine.clear_graph()
        assert len(engine.db.nodes) == 0


def test_benchmark_single_engine():
    """Verify benchmarking runner on individual graph engines."""
    nodes, edges = generate_benchmark_dataset(num_nodes=20, num_edges=30)
    runner = GraphEngineBenchmark(force_mock=True)

    with KuzuEngine(force_mock=True) as engine:
        res = runner.benchmark_engine("KuzuEngine", engine, nodes, edges, num_read_iterations=5)

        assert res["engine"] == "KuzuEngine"
        assert res["node_write"]["count"] == 20
        assert res["node_write"]["throughput_nodes_per_sec"] > 0
        assert res["edge_write"]["count"] == 30
        assert res["edge_write"]["throughput_edges_per_sec"] > 0
        assert res["read_1hop"]["iterations"] == 5
        assert res["read_2hop"]["iterations"] == 5


def test_run_graph_engine_benchmark_full():
    """Verify full benchmark execution across Neo4jClient, KuzuEngine, and AgeEngine."""
    report = run_graph_engine_benchmark(
        engines=["Neo4jClient", "KuzuEngine", "AgeEngine"],
        num_nodes=30,
        num_edges=50,
        num_read_iterations=5,
        force_mock=True,
    )

    assert "timestamp" in report
    assert report["config"]["num_nodes"] == 30
    assert report["config"]["num_edges"] == 50
    assert report["config"]["num_read_iterations"] == 5

    engines = report["engines"]
    assert "Neo4jClient" in engines
    assert "KuzuEngine" in engines
    assert "AgeEngine" in engines

    for name in ["Neo4jClient", "KuzuEngine", "AgeEngine"]:
        metrics = engines[name]
        assert metrics["node_write"]["count"] == 30
        assert metrics["edge_write"]["count"] == 50
        assert "throughput_nodes_per_sec" in metrics["node_write"]
        assert "throughput_edges_per_sec" in metrics["edge_write"]
        assert "mean_latency_ms" in metrics["read_1hop"]
        assert "mean_latency_ms" in metrics["read_2hop"]

    summary = report["summary"]
    assert summary["fastest_node_write"] in engines
    assert summary["fastest_edge_write"] in engines
    assert summary["lowest_1hop_latency"] in engines
    assert summary["lowest_2hop_latency"] in engines


def test_run_graph_engine_benchmark_file_output(tmp_path):
    """Verify JSON benchmark report is saved to file when output_path is provided."""
    output_file = tmp_path / "benchmark_report.json"
    report = run_graph_engine_benchmark(
        engines=["KuzuEngine", "AgeEngine"],
        num_nodes=10,
        num_edges=15,
        num_read_iterations=3,
        output_path=str(output_file),
        force_mock=True,
    )

    assert output_file.exists()
    with open(output_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["timestamp"] == report["timestamp"]
    assert "KuzuEngine" in loaded["engines"]
    assert "AgeEngine" in loaded["engines"]
