import json
import os
import uuid
import pytest
from unittest.mock import MagicMock, patch
from app.discovery.lightrag_runner import DiscoveryClusteringRunner
from app.discovery.harvest import DiscoveryHarvester
from app.discovery.cli import main as cli_main

class MockCursor:
    def __init__(self, entities, relationships):
        self.entities = entities
        self.relationships = relationships
        self.execute_calls = []
        self.inserted_relationships = []
        self.current_query = ""

    def execute(self, query: str, params: tuple | None = None) -> None:
        self.execute_calls.append((query, params))
        self.current_query = query
        if "INSERT INTO relationships" in query:
            self.inserted_relationships.append(params)

    def fetchall(self) -> list:
        if "FROM normalized_entities" in self.current_query:
            return [(e["id"], e["canonical_id"], e["name"], e["entity_type"]) for e in self.entities]
        elif "FROM relationships" in self.current_query:
            return [(r["source_entity_id"], r["target_entity_id"], r.get("confidence_score", 1.0)) for r in self.relationships]
        return []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class MockConnection:
    def __init__(self, entities, relationships):
        self.cursor_obj = MockCursor(entities, relationships)
        self.committed = False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.committed = True

    def close(self):
        pass

class MockNeo4jRecord:
    def __init__(self, data):
        self.data = data
    def get(self, key):
        return self.data.get(key)
    def __getitem__(self, key):
        return self.data[key]
    def keys(self):
        return self.data.keys()

class MockNeo4jClient:
    def __init__(self, existing_edges=None):
        self.existing_edges = existing_edges or []
        self.queries = []

    def execute_query(self, query, parameters=None):
        self.queries.append((query, parameters))
        if "MATCH (a:Entity)-[r]->(b:Entity)" in query:
            return [MockNeo4jRecord({"source": src, "target": tgt}) for src, tgt in self.existing_edges]
        return []

    def close(self):
        pass

@pytest.fixture
def sample_data():
    e1 = str(uuid.uuid4())
    e2 = str(uuid.uuid4())
    e3 = str(uuid.uuid4())
    e4 = str(uuid.uuid4())
    e_ignored = str(uuid.uuid4())

    entities = [
        {"id": e1, "canonical_id": "G1", "name": "Gene1", "entity_type": "GENE"},
        {"id": e2, "canonical_id": "G2", "name": "Gene2", "entity_type": "GENE"},
        {"id": e3, "canonical_id": "D1", "name": "Disease1", "entity_type": "DISEASE"},
        {"id": e4, "canonical_id": "C1", "name": "CellType1", "entity_type": "CELL_TYPE"},
        {"id": e_ignored, "canonical_id": "X1", "name": "Ignored", "entity_type": "OTHER"}
    ]

    relationships = [
        {"source_entity_id": e1, "target_entity_id": e2, "confidence_score": 0.9},
        {"source_entity_id": e2, "target_entity_id": e3, "confidence_score": 0.8},
        {"source_entity_id": e3, "target_entity_id": e_ignored, "confidence_score": 0.95}
    ]

    return entities, relationships

def test_clustering_connected_components(sample_data):
    entities, relationships = sample_data
    conn = MockConnection(entities, relationships)
    runner = DiscoveryClusteringRunner(conn)

    communities = runner.run_clustering(min_confidence=0.5, algorithm="connected_components")
    
    assert len(communities) == 2
    
    c1 = communities[0]
    assert len(c1) == 3
    c1_cids = [x["canonical_id"] for x in c1]
    assert "G1" in c1_cids
    assert "G2" in c1_cids
    assert "D1" in c1_cids

    c2 = communities[1]
    assert len(c2) == 1
    assert c2[0]["canonical_id"] == "C1"

def test_clustering_label_propagation(sample_data):
    entities, relationships = sample_data
    conn = MockConnection(entities, relationships)
    runner = DiscoveryClusteringRunner(conn)

    communities = runner.run_clustering(min_confidence=0.5, algorithm="label_propagation")
    assert len(communities) >= 1

def test_clustering_runner_min_confidence(sample_data):
    entities, relationships = sample_data
    conn = MockConnection(entities, relationships)
    runner = DiscoveryClusteringRunner(conn)

    communities = runner.run_clustering(min_confidence=0.85, algorithm="connected_components")
    assert len(communities) == 3
    assert len(communities[0]) == 2
    assert communities[0][0]["canonical_id"] == "G1"
    assert communities[0][1]["canonical_id"] == "G2"

def test_harvester_basic(sample_data, tmp_path):
    entities, relationships = sample_data
    communities = [[entities[0], entities[1], entities[2]]]
    
    conn = MockConnection(entities, relationships)
    neo4j = MockNeo4jClient()
    
    harvester = DiscoveryHarvester(conn, neo4j)
    export_path = os.path.join(tmp_path, "candidates.json")
    
    candidates = harvester.harvest_candidates(
        communities=communities,
        export_json_path=export_path,
        insert_to_db=True,
        limit=10
    )

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand["source_entity"]["canonical_id"] in ["G1", "D1"]
    assert cand["target_entity"]["canonical_id"] in ["G1", "D1"]
    assert cand["discovery_score"] > 0.0

    assert len(conn.cursor().inserted_relationships) == 1
    inserted = conn.cursor().inserted_relationships[0]
    assert inserted[5] == "PENDING"
    assert inserted[6] == "automated_discovery"

    assert os.path.exists(export_path)
    with open(export_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["source_entity"]["canonical_id"] in ["G1", "D1"]
    assert data[0]["target_entity"]["canonical_id"] in ["G1", "D1"]

def test_harvester_skips_existing_neo4j(sample_data):
    entities, relationships = sample_data
    communities = [[entities[0], entities[1], entities[2]]]
    conn = MockConnection(entities, relationships)
    
    neo4j = MockNeo4jClient(existing_edges=[("G1", "D1")])
    
    harvester = DiscoveryHarvester(conn, neo4j)
    candidates = harvester.harvest_candidates(
        communities=communities,
        insert_to_db=False
    )
    
    assert len(candidates) == 0

def test_cli_execution(sample_data):
    entities, relationships = sample_data
    conn = MockConnection(entities, relationships)
    neo4j = MockNeo4jClient()

    with patch("psycopg.connect", return_value=conn),          patch("app.discovery.cli.Neo4jClient", return_value=neo4j):
        
        args = ["--pg-dsn", "fake_dsn", "--no-insert-db"]
        with patch("sys.argv", ["cli.py"] + args):
            cli_main()
