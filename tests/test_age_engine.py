"""Unit test suite for AgeEngine in app/materialization/age_engine.py."""

import pytest

from app.materialization.age_engine import (
    AgeEngine,
    MockAgeConnection,
    MockAgeDatabase,
    MockAgeQueryResult,
)


@pytest.fixture
def engine():
    """Fixture providing AgeEngine instance in mock mode."""
    eng = AgeEngine(graph_name="ebv_graph", force_mock=True)
    eng.init_schema()
    yield eng
    eng.close()


def test_imports_and_mock_fallback():
    """Verify fallback mechanism and mock instances."""
    mock_eng = AgeEngine(graph_name="ebv_graph", force_mock=True)
    assert mock_eng.is_mock is True
    assert isinstance(mock_eng.db, MockAgeDatabase)
    assert isinstance(mock_eng.conn, MockAgeConnection)
    mock_eng.close()


def test_mock_result_set():
    """Verify MockAgeQueryResult fetch methods."""
    rows = [{"id": "1", "name": "EBNA1"}, {"id": "2", "name": "LMP1"}]
    res = MockAgeQueryResult(rows=rows, column_names=["id", "name"])

    assert res.get_column_names() == ["id", "name"]
    assert res.has_next() is True
    first = res.fetchone()
    assert first == {"id": "1", "name": "EBNA1"}

    next_row_cols = res.get_next()
    assert next_row_cols == ["2", "LMP1"]
    assert res.has_next() is False

    with pytest.raises(StopIteration):
        res.get_next()

    all_rows = res.fetchall()
    assert len(all_rows) == 2
    assert res.rows_as_dict() == rows


def test_init_schema(engine):
    """Verify schema creation queries for Apache AGE graph 'ebv_graph'."""
    queries = engine.init_schema()
    assert len(queries) == 5
    assert any("create_graph" in q and "ebv_graph" in q for q in queries)
    assert any("Entity" in q for q in queries)
    assert any("Paper" in q for q in queries)
    assert any("ASSOCIATED_WITH" in q for q in queries)
    assert any("IS_MARKER_FOR" in q for q in queries)


def test_execute_query_and_cypher(engine):
    """Verify parameterized Cypher execution via execute_cypher and execute_query."""
    res = engine.execute_cypher(
        "CREATE (e:Entity {canonical_id: $id, name: $name, entity_type: $type}) RETURN e",
        {"id": "HGNC:1234", "name": "EBNA1", "type": "Gene"},
    )
    assert isinstance(res, list)
    if res:
        assert res[0].get("id") == "HGNC:1234"
        assert res[0].get("name") == "EBNA1"

    res_alias = engine.execute_query(
        "MATCH (e:Entity) RETURN e.canonical_id AS id, e.name AS name",
        {"entity_id": "HGNC:1234"},
    )
    assert isinstance(res_alias, list)


def test_bulk_upsert_nodes(engine):
    """Verify bulk node upserts for Entity and Paper labels."""
    entity_nodes = [
        {
            "canonical_id": "HGNC:11989",
            "name": "EBNA1",
            "entity_type": "ViralProtein",
            "ontology_source": "UniProt",
            "synonyms": ["EBNA-1", "BKRF1"],
        },
        {
            "canonical_id": "HGNC:6646",
            "name": "LMP1",
            "entity_type": "ViralProtein",
            "ontology_source": "UniProt",
            "synonyms": ["LMP-1", "BNRF1"],
        },
    ]
    cnt_entities = engine.bulk_upsert_nodes("Entity", entity_nodes)
    assert cnt_entities == 2

    paper_nodes = [
        {
            "doi": "10.1038/s41586-020-0001",
            "pmid": "32000001",
            "title": "Epstein-Barr Virus Latency Mechanics",
            "journal": "Nature",
            "publication_year": 2020,
        }
    ]
    cnt_papers = engine.bulk_upsert_nodes("Paper", paper_nodes)
    assert cnt_papers == 1

    res = engine.execute_cypher(
        "MATCH (e:Entity) RETURN e.canonical_id AS id, e.name AS name"
    )
    retrieved_ids = {r.get("id") for r in res}
    assert "HGNC:11989" in retrieved_ids
    assert "HGNC:6646" in retrieved_ids


def test_bulk_upsert_relationships(engine):
    """Verify bulk relationship upserts for ASSOCIATED_WITH and IS_MARKER_FOR."""
    entities = [
        {"canonical_id": "HGNC:1", "name": "EBNA1", "entity_type": "Protein"},
        {"canonical_id": "HGNC:2", "name": "LMP1", "entity_type": "Protein"},
        {"canonical_id": "HGNC:3", "name": "CD21", "entity_type": "Receptor"},
    ]
    engine.bulk_upsert_nodes("Entity", entities)

    assoc_edges = [
        {
            "source_id": "HGNC:1",
            "target_id": "HGNC:2",
            "relationship_type": "INTERACTS_WITH",
            "confidence": 0.95,
            "evidence_text": "EBNA1 and LMP1 co-occur in latent infections.",
            "curation_status": "APPROVED",
        }
    ]
    cnt_assoc = engine.bulk_upsert_edges("ASSOCIATED_WITH", assoc_edges)
    assert cnt_assoc == 1

    marker_edges = [
        {
            "source_id": "HGNC:2",
            "target_id": "HGNC:3",
            "log2_fold_change": 3.4,
            "p_value": 0.0001,
            "cell_type": "B Cell",
            "confidence": 0.88,
        }
    ]
    cnt_marker = engine.bulk_upsert_relationships("IS_MARKER_FOR", marker_edges)
    assert cnt_marker == 1


def test_2hop_neighborhood_retrieval(engine):
    """Verify 2-hop neighborhood path traversal and structured output."""
    entities = [
        {"canonical_id": "HGNC:100", "name": "EBNA1", "entity_type": "Protein"},
        {"canonical_id": "HGNC:200", "name": "LMP1", "entity_type": "Protein"},
        {"canonical_id": "HGNC:300", "name": "CD21", "entity_type": "Receptor"},
    ]
    engine.bulk_upsert_nodes("Entity", entities)

    edges1 = [
        {
            "source_id": "HGNC:100",
            "target_id": "HGNC:200",
            "relationship_type": "BINDS",
            "confidence": 0.90,
        }
    ]
    engine.bulk_upsert_edges("ASSOCIATED_WITH", edges1)

    edges2 = [
        {
            "source_id": "HGNC:200",
            "target_id": "HGNC:300",
            "log2_fold_change": 2.1,
            "p_value": 0.005,
            "confidence": 0.85,
        }
    ]
    engine.bulk_upsert_edges("IS_MARKER_FOR", edges2)

    neighborhood = engine.get_2hop_neighborhood("HGNC:100")
    assert neighborhood["start_id"] == "HGNC:100"

    hop1_ids = {n["canonical_id"] for n in neighborhood["hop1_nodes"]}
    assert "HGNC:200" in hop1_ids

    hop2_ids = {n["canonical_id"] for n in neighborhood["hop2_nodes"]}
    assert "HGNC:300" in hop2_ids

    paths = engine.get_neighborhood_paths("HGNC:100", max_hops=2)
    assert len(paths) >= 1
    assert any(p["start"] == "HGNC:100" for p in paths)


def test_clear_graph(engine):
    """Verify clearing graph resets graph contents."""
    entities = [{"canonical_id": "HGNC:999", "name": "TempNode", "entity_type": "Gene"}]
    engine.bulk_upsert_nodes("Entity", entities)

    res_before = engine.execute_cypher("MATCH (e:Entity) RETURN e.canonical_id AS id")
    assert len(res_before) >= 1

    engine.clear_graph()

    res_after = engine.execute_cypher("MATCH (e:Entity) RETURN e.canonical_id AS id")
    assert len(res_after) == 0


def test_context_manager():
    """Verify context manager interface __enter__ and __exit__."""
    with AgeEngine(graph_name="ebv_graph", force_mock=True) as eng:
        eng.init_schema()
        assert eng.is_mock is True
        res = eng.execute_cypher(
            "CREATE (e:Entity {canonical_id: 'E_CTX', name: 'ContextNode'}) RETURN e"
        )
        assert isinstance(res, list)


def test_custom_graph_name():
    """Verify AgeEngine with a custom graph name."""
    with AgeEngine(graph_name="custom_ebv_graph", force_mock=True) as custom_eng:
        queries = custom_eng.init_schema()
        assert any("custom_ebv_graph" in q for q in queries)

        custom_eng.bulk_upsert_nodes(
            "Entity",
            [{"canonical_id": "CUST:1", "name": "CustomNode", "entity_type": "Gene"}],
        )

        res = custom_eng.execute_cypher("MATCH (e:Entity) RETURN e")
        assert len(res) == 1
        assert custom_eng.get_2hop_neighborhood("CUST:1")["start_id"] == "CUST:1"
