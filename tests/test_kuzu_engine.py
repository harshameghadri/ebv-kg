"""Unit test suite for KuzuEngine in app/materialization/kuzu_engine.py."""

import pytest

from app.materialization.kuzu_engine import (
    KuzuEngine,
    KUZU_AVAILABLE,
    MockKuzuDatabase,
    MockKuzuConnection,
    MockQueryResult,
)


@pytest.fixture(params=[False, True] if KUZU_AVAILABLE else [True])
def engine(request):
    """Fixture providing KuzuEngine instance (testing both native C++ kuzu and mock engine when available)."""
    eng = KuzuEngine(db_path=":memory:", force_mock=request.param)
    eng.init_schema()
    yield eng
    eng.close()


def test_imports_and_mock_fallback():
    """Verify fallback mechanism and import flags."""
    mock_eng = KuzuEngine(db_path=":memory:", force_mock=True)
    assert mock_eng.is_mock is True
    assert isinstance(mock_eng.db, MockKuzuDatabase)
    assert isinstance(mock_eng.conn, MockKuzuConnection)
    mock_eng.close()


def test_init_schema(engine):
    """Verify schema creation for Entity, Paper, ASSOCIATED_WITH, IS_MARKER_FOR."""
    queries = engine.init_schema()
    assert len(queries) == 4
    assert any("Entity" in q for q in queries)
    assert any("Paper" in q for q in queries)
    assert any("ASSOCIATED_WITH" in q for q in queries)
    assert any("IS_MARKER_FOR" in q for q in queries)


def test_execute_query(engine):
    """Verify parameterized Cypher execution."""
    res = engine.execute_query(
        "CREATE (e:Entity {canonical_id: $id, name: $name, entity_type: $type}) RETURN e.canonical_id AS id, e.name AS name",
        {"id": "HGNC:1234", "name": "EBNA1", "type": "Gene"},
    )
    assert isinstance(res, list)
    if res:
        assert res[0].get("id") == "HGNC:1234"
        assert res[0].get("name") == "EBNA1"


def test_bulk_upsert_nodes(engine):
    """Verify bulk node upserts for Entity and Paper tables."""
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

    res = engine.execute_query("MATCH (e:Entity) RETURN e.canonical_id AS id, e.name AS name")
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
        {"source_id": "HGNC:100", "target_id": "HGNC:200", "relationship_type": "BINDS", "confidence": 0.90}
    ]
    engine.bulk_upsert_edges("ASSOCIATED_WITH", edges1)

    edges2 = [
        {"source_id": "HGNC:200", "target_id": "HGNC:300", "log2_fold_change": 2.1, "p_value": 0.005, "confidence": 0.85}
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
    """Verify clearing all data from graph engine."""
    entities = [{"canonical_id": "HGNC:999", "name": "TempNode", "entity_type": "Gene"}]
    engine.bulk_upsert_nodes("Entity", entities)

    res_before = engine.execute_query("MATCH (e:Entity) RETURN e.canonical_id AS id")
    assert len(res_before) >= 1

    engine.clear_graph()

    res_after = engine.execute_query("MATCH (e:Entity) RETURN e.canonical_id AS id")
    assert len(res_after) == 0


def test_context_manager():
    """Verify context manager interface __enter__ and __exit__."""
    with KuzuEngine(db_path=":memory:", force_mock=True) as eng:
        eng.init_schema()
        res = eng.execute_query("CREATE (e:Entity {canonical_id: 'E_CTX', name: 'ContextNode'}) RETURN e")
        assert eng.is_mock is True
