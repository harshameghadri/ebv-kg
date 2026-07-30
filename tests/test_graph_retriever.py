"""Unit tests for the GraphRetriever class and retrieve_graph_context function."""

import pytest
from unittest.mock import MagicMock, call

from app.retrieval.graph import GraphRetriever, retrieve_graph_context


@pytest.fixture
def mock_neo4j_client():
    """Fixture providing a mock Neo4jClient."""
    client = MagicMock()
    return client


def test_find_entities_by_name(mock_neo4j_client):
    """Verify that find_entities_by_name queries Neo4j for matches by name or synonym."""
    retriever = GraphRetriever(neo4j_client=mock_neo4j_client)

    # Return record objects or dicts
    mock_neo4j_client.execute_query.return_value = [
        {"canonical_id": "HGNC:1"}
    ]

    res = retriever._find_entities_by_name("TP53")
    assert res == ["HGNC:1"]

    # Verify Cypher query targets both name and synonyms, using parameterization
    mock_neo4j_client.execute_query.assert_called_once()
    cypher_call = mock_neo4j_client.execute_query.call_args[0][0]
    params = mock_neo4j_client.execute_query.call_args[0][1]

    assert "toLower(e.name) = toLower($term)" in cypher_call
    assert "any(syn in e.synonyms WHERE toLower(syn) = toLower($term))" in cypher_call
    assert params == {"term": "TP53"}


def test_extract_candidates_simple_keyword(mock_neo4j_client):
    """Verify keyword extraction uses word-boundary search when NER/SynonymResolver are absent.

    Should extract exact words/phrases but not arbitrary substring overlaps.
    """
    retriever = GraphRetriever(neo4j_client=mock_neo4j_client)

    # Mock Neo4j return for all known entities
    mock_neo4j_client.execute_query.return_value = [
        {"canonical_id": "HGNC:1", "name": "LMP1", "synonyms": ["latent membrane protein 1"]},
        {"canonical_id": "HGNC:2", "name": "EBV", "synonyms": []},
        {"canonical_id": "HGNC:3", "name": "cell", "synonyms": []},  # Substring check
    ]

    # Query with exact boundaries
    candidates = retriever.extract_candidates("LMP1 induces cell signaling in EBV-infected systems.")

    # "LMP1", "EBV", and "cell" should match because they are separate words (EBV-infected contains word boundary)
    assert set(candidates) == {"HGNC:1", "HGNC:2", "HGNC:3"}

    # Query with substring but no boundary match
    # E.g. "EBVinfected" has no boundary, "LMP10" is a different gene
    mock_neo4j_client.execute_query.reset_mock()
    candidates_no_match = retriever.extract_candidates("EBVinfected cell lines with LMP10")
    # should match "cell" but NOT "HGNC:1" (LMP1) or "HGNC:2" (EBV)
    assert "HGNC:1" not in candidates_no_match
    assert "HGNC:2" not in candidates_no_match
    assert "HGNC:3" in candidates_no_match


def test_extract_candidates_with_ner_resolver(mock_neo4j_client):
    """Verify that when NER & SynonymResolver are provided, candidate extraction delegates to them."""
    mock_ner = MagicMock()
    mock_ner.extract.return_value = [
        {"text": "latent membrane protein 1", "entity_type": "PROTEIN"}
    ]

    mock_resolver = MagicMock()
    mock_resolver.resolve.return_value = {
        "canonical_id": "HGNC:1",
        "symbol": "LMP1",
    }

    retriever = GraphRetriever(
        neo4j_client=mock_neo4j_client,
        ner_extractor=mock_ner,
        synonym_resolver=mock_resolver,
    )

    candidates = retriever.extract_candidates("A study on latent membrane protein 1.")

    assert candidates == ["HGNC:1"]
    mock_ner.extract.assert_called_once_with("A study on latent membrane protein 1.")
    mock_resolver.resolve.assert_called_once_with(
        "latent membrane protein 1", category="PROTEIN"
    )
    # Simple keyword search fallback should NOT be called since NER found a candidate
    mock_neo4j_client.execute_query.assert_not_called()


def test_get_neighborhood_hops_and_papers(mock_neo4j_client):
    """Verify 2-hop traversal logic and paper querying.

    Ensures that:
    1. Hop 1 runs using the start IDs.
    2. Hop 2 runs using all discovered IDs from hop 1.
    3. Start IDs are populated as fallback Entity nodes even if they lack edges.
    4. Papers are fetched only for the final complete set of neighborhood entities.
    """
    retriever = GraphRetriever(neo4j_client=mock_neo4j_client, min_confidence=0.75)

    # 1. Configure mock responses for Neo4jClient
    # First call: Hop 1 relationships
    hop1_records = [
        {
            "source_id": "HGNC:1",
            "source_name": "LMP1",
            "source_type": "PROTEIN",
            "rel_type": "INTERACTS_WITH",
            "confidence_score": 0.85,
            "curation_status": "APPROVED",
            "rel_id": "r1",
            "target_id": "HGNC:2",
            "target_name": "JAK3",
            "target_type": "GENE",
        }
    ]

    # Second call: Hop 2 relationships (starts with HGNC:1 + HGNC:2)
    hop2_records = [
        # Relationship 1 again
        {
            "source_id": "HGNC:1",
            "source_name": "LMP1",
            "source_type": "PROTEIN",
            "rel_type": "INTERACTS_WITH",
            "confidence_score": 0.85,
            "curation_status": "APPROVED",
            "rel_id": "r1",
            "target_id": "HGNC:2",
            "target_name": "JAK3",
            "target_type": "GENE",
        },
        # Relationship 2: HGNC:2 interacting with HGNC:3
        {
            "source_id": "HGNC:2",
            "source_name": "JAK3",
            "source_type": "GENE",
            "rel_type": "TARGETS",
            "confidence_score": 0.90,
            "curation_status": "APPROVED",
            "rel_id": "r2",
            "target_id": "HGNC:3",
            "target_name": "STAT3",
            "target_type": "GENE",
        },
    ]

    # Third call: Start entities fallback details
    start_entity_details = [
        {"canonical_id": "HGNC:1", "name": "LMP1", "entity_type": "PROTEIN"}
    ]

    # Fourth call: Papers matching the full neighborhood ("HGNC:1", "HGNC:2", "HGNC:3")
    paper_records = [
        {
            "doi": "10.1038/s41586-020-2012-7",
            "pmid": "32132123",
            "title": "EBV Study",
            "journal": "Nature",
            "published_date": "2020-03-04",
            "entity_id": "HGNC:1",
            "confidence_score": 0.95,
        },
        {
            "doi": "10.1038/s41586-020-2012-7",
            "pmid": "32132123",
            "title": "EBV Study",
            "journal": "Nature",
            "published_date": "2020-03-04",
            "entity_id": "HGNC:2",
            "confidence_score": 0.85,
        },
    ]

    def mock_query_router(query_str, params=None):
        if "IN $entity_ids" in query_str and "type(r) <> \"MENTIONS\"" in query_str:
            if params.get("entity_ids") == ["HGNC:1"]:
                return hop1_records
            else:
                # Hop 2 receives ["HGNC:1", "HGNC:2"] (order may vary, set comparison)
                assert set(params.get("entity_ids")) == {"HGNC:1", "HGNC:2"}
                return hop2_records
        elif "MATCH (e:Entity) WHERE e.canonical_id IN $entity_ids" in query_str:
            return start_entity_details
        elif "MATCH (p:Paper)-[m:MENTIONS]->(e:Entity)" in query_str:
            assert set(params.get("entity_ids")) == {"HGNC:1", "HGNC:2", "HGNC:3"}
            return paper_records
        return []

    mock_neo4j_client.execute_query.side_effect = mock_query_router

    # Run get_neighborhood
    res = retriever.get_neighborhood(["HGNC:1"])

    # Verify return structures and deduplication
    assert len(res["entities"]) == 3
    assert {e["canonical_id"] for e in res["entities"]} == {"HGNC:1", "HGNC:2", "HGNC:3"}

    assert len(res["relationships"]) == 2
    assert {r["id"] for r in res["relationships"]} == {"r1", "r2"}

    assert len(res["papers"]) == 1  # 2 records for same paper DOI -> deduplicated
    assert res["papers"][0]["doi"] == "10.1038/s41586-020-2012-7"

    assert len(res["mentions"]) == 2


def test_format_neighborhood_context(mock_neo4j_client):
    """Verify context formatting matches specified structure, including co-mentions and individual mentions."""
    retriever = GraphRetriever(neo4j_client=mock_neo4j_client)

    neighborhood = {
        "entities": [
            {"canonical_id": "HGNC:1", "name": "LMP1", "entity_type": "PROTEIN"},
            {"canonical_id": "HGNC:2", "name": "JAK3", "entity_type": "GENE"},
            {"canonical_id": "HGNC:3", "name": "STAT3", "entity_type": "GENE"},
        ],
        "relationships": [
            {
                "id": "r1",
                "source_id": "HGNC:1",
                "source_name": "LMP1",
                "source_type": "PROTEIN",
                "target_id": "HGNC:2",
                "target_name": "JAK3",
                "target_type": "GENE",
                "rel_type": "INTERACTS_WITH",
                "confidence_score": 0.85,
            },
            {
                "id": "r2",
                "source_id": "HGNC:2",
                "source_name": "JAK3",
                "source_type": "GENE",
                "target_id": "HGNC:3",
                "target_name": "STAT3",
                "target_type": "GENE",
                "rel_type": "TARGETS",
                "confidence_score": 0.90,
            },
        ],
        "papers": [
            {
                "doi": "10.1038/s41586-020-2012-7",
                "pmid": "32132123",
                "title": "EBV Study",
            }
        ],
        "mentions": [
            {"paper_doi": "10.1038/s41586-020-2012-7", "entity_id": "HGNC:1", "confidence_score": 0.95},
            {"paper_doi": "10.1038/s41586-020-2012-7", "entity_id": "HGNC:2", "confidence_score": 0.85},
        ],
    }

    formatted = retriever.format_neighborhood_context(neighborhood)

    # Check formatting of co-mentioned relationship (r1 has both HGNC:1 and HGNC:2 in paper)
    assert "- LMP1 (PROTEIN) interacts with JAK3 (GENE) [confidence: 0.85] in Paper (DOI: 10.1038/s41586-020-2012-7, PMID: 32132123)." in formatted

    # Check formatting of relationship without co-mention (r2 has target HGNC:3 which is not in paper)
    assert "- JAK3 (GENE) targets STAT3 (GENE) [confidence: 0.90]." in formatted

    # Check formatting of literature mentions block
    assert "Entity Literature Mentions:" in formatted
    assert "- LMP1 (PROTEIN) is mentioned in: 'EBV Study' (DOI: 10.1038/s41586-020-2012-7, PMID: 32132123)" in formatted
    assert "- JAK3 (GENE) is mentioned in: 'EBV Study' (DOI: 10.1038/s41586-020-2012-7, PMID: 32132123)" in formatted
    # STAT3 has no mentions, so it should not appear in mentions block
    assert "STAT3 (GENE) is mentioned in" not in formatted


def test_retrieve_graph_context_helper(mock_neo4j_client):
    """Verify that retrieve_graph_context helper works at module-level."""
    # Mock return values to trace full pipeline execution
    mock_neo4j_client.execute_query.side_effect = [
        # 1. extract_candidates (fetch all entities)
        [{"canonical_id": "HGNC:1", "name": "LMP1", "synonyms": []}],
        # 2. get_neighborhood: hop 1 relations
        [],
        # 3. get_neighborhood: start entity details
        [{"canonical_id": "HGNC:1", "name": "LMP1", "entity_type": "PROTEIN"}],
    ]

    res = retrieve_graph_context(query="LMP1 research", neo4j_client=mock_neo4j_client)

    # Just start entity details with no connections
    assert "Entity Literature Mentions:" not in res
    assert "LMP1 (PROTEIN)" in res
