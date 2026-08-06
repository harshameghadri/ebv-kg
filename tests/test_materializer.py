"""Unit tests for the Materializer class."""

import datetime
from typing import Any
from unittest.mock import MagicMock, call

import pytest

from app.materialization.materializer import Materializer


class MockCursor:
    """Mock database cursor that returns query-specific results."""

    def __init__(self, results: dict) -> None:
        self.results = results
        self.execute_calls = []
        self.current_query = ""

    def execute(self, query: str, params: list | None = None) -> None:
        self.execute_calls.append((query, params))
        self.current_query = query

    def fetchall(self) -> list:
        if "FROM relationships" in self.current_query:
            return self.results.get("relationships", [])
        elif "relationship_evidence" in self.current_query or "ev.confidence_score" in self.current_query:
            return self.results.get("relationship_evidence", [])
        elif "FROM normalized_entities" in self.current_query:
            return self.results.get("normalized_entities", [])
        elif "FROM documents" in self.current_query:
            return self.results.get("documents", [])
        return []


    def __enter__(self) -> "MockCursor":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


class MockConnection:
    """Mock database connection."""

    def __init__(self, results: dict) -> None:
        self.cursor_obj = MockCursor(results)

    def cursor(self, row_factory: Any = None) -> MockCursor:
        return self.cursor_obj


@pytest.fixture
def mock_neo4j_client():
    """Fixture providing a mock Neo4j client."""
    client = MagicMock()
    # Configure mock responses for upsert methods
    client.bulk_upsert_nodes.side_effect = lambda label, nodes, id_property=None: len(
        nodes
    )

    def mock_upsert_edges(
        rel_type,
        edges,
        source_label=None,
        target_label=None,
        source_key=None,
        target_key=None,
    ):
        return len(edges)

    client.bulk_upsert_edges.side_effect = mock_upsert_edges
    return client


def test_init_schema(mock_neo4j_client):
    """Verify init_schema calls Neo4jClient's init_schema."""
    materializer = Materializer(neo4j_client=mock_neo4j_client)
    mock_neo4j_client.init_schema.return_value = ["CONSTRAINT 1"]

    res = materializer.init_schema()
    assert res == ["CONSTRAINT 1"]
    mock_neo4j_client.init_schema.assert_called_once()


def test_clear_graph(mock_neo4j_client):
    """Verify clear_graph calls Neo4jClient's clear_graph."""
    materializer = Materializer(neo4j_client=mock_neo4j_client)

    materializer.clear_graph()
    mock_neo4j_client.clear_graph.assert_called_once()


def test_materialize_graph_basic(mock_neo4j_client):
    """Verify standard materialization pipeline behavior with mock DB tables."""
    fake_entities = [
        {
            "id": "e302ab8f-12e0-4a87-9bb3-585e4933eb91",
            "canonical_id": "HGNC:11985",
            "name": "TP53",
            "entity_type": "GENE",
            "ontology_source": "HGNC",
            "synonyms": ["p53", "tumor protein p53"],
        },
        {
            "id": "e302ab8f-12e0-4a87-9bb3-585e4933eb92",
            "canonical_id": "HGNC:672",
            "name": "BRCA2",
            "entity_type": "GENE",
            "ontology_source": "HGNC",
            "synonyms": None,  # Test handling None synonyms
        },
    ]

    fake_papers = [
        {
            "id": "p7248efd-88b9-4a0b-968b-5777dfb8eb22",
            "doi": "10.1038/s41586-020-2012-7",
            "pmid": "32132123",
            "title": "EBV Study",
            "journal": "Nature",
            "published_date": datetime.date(2020, 3, 4),
        },
        {
            "id": "p7248efd-88b9-4a0b-968b-5777dfb8eb23",
            "doi": None,  # Lacks DOI, should be skipped
            "pmid": "1111111",
            "title": "No DOI Study",
            "journal": "Science",
            "published_date": None,
        },
    ]

    fake_relationships = [
        {
            "id": "r1004ab8-99e0-4c87-9ab3-585e4933eb99",
            "relationship_type": "ASSOCIATED_WITH",
            "confidence_score": 0.85,
            "curation_status": "APPROVED",
            "source_type": "NER",
            "source_canonical_id": "HGNC:11985",
            "target_canonical_id": "HGNC:672",
            "evidence_count": 1,
            "source_pmids": ["32132123"],
            "source_dois": ["10.1038/s41586-020-2012-7"],
        }
    ]


    fake_mentions = [
        {
            "source_doi": "10.1038/s41586-020-2012-7",
            "target_canonical_id": "HGNC:11985",
            "confidence_score": 0.9,
        }
    ]

    db_results = {
        "normalized_entities": fake_entities,
        "documents": fake_papers,
        "relationships": fake_relationships,
        "relationship_evidence": fake_mentions,
    }

    mock_conn = MockConnection(db_results)
    materializer = Materializer(neo4j_client=mock_neo4j_client)

    stats = materializer.materialize_graph(pg_conn=mock_conn)

    # Check stats output
    assert stats["entities"] == 2
    assert stats["papers"] == 1  # 1 paper has DOI, 1 skipped
    assert stats["relationships"] == 1
    assert stats["mentions"] == 1

    # Check PostgreSQL queries executed
    cursor = mock_conn.cursor()
    assert len(cursor.execute_calls) == 4

    assert "normalized_entities" in cursor.execute_calls[0][0]
    assert "documents" in cursor.execute_calls[1][0]
    assert "relationships" in cursor.execute_calls[2][0]
    assert "relationship_evidence" in cursor.execute_calls[3][0]

    # Verify Neo4jClient bulk upsert node calls
    mock_neo4j_client.bulk_upsert_nodes.assert_has_calls(
        [
            call(
                label="Entity",
                nodes=[
                    {
                        "id": "e302ab8f-12e0-4a87-9bb3-585e4933eb91",
                        "canonical_id": "HGNC:11985",
                        "name": "TP53",
                        "entity_type": "GENE",
                        "ontology_source": "HGNC",
                        "synonyms": ["p53", "tumor protein p53"],
                    },
                    {
                        "id": "e302ab8f-12e0-4a87-9bb3-585e4933eb92",
                        "canonical_id": "HGNC:672",
                        "name": "BRCA2",
                        "entity_type": "GENE",
                        "ontology_source": "HGNC",
                        "synonyms": [],
                    },
                ],
                id_property="canonical_id",
            ),
            call(
                label="Paper",
                nodes=[
                    {
                        "id": "p7248efd-88b9-4a0b-968b-5777dfb8eb22",
                        "doi": "10.1038/s41586-020-2012-7",
                        "pmid": "32132123",
                        "title": "EBV Study",
                        "journal": "Nature",
                        "published_date": "2020-03-04",
                    }
                ],
                id_property="doi",
            ),
        ],
        any_order=False,
    )

    # Verify Neo4jClient bulk upsert edge calls
    mock_neo4j_client.bulk_upsert_edges.assert_has_calls(
        [
            call(
                rel_type="ASSOCIATED_WITH",
                edges=[
                    {
                        "id": "r1004ab8-99e0-4c87-9ab3-585e4933eb99",
                        "source_canonical_id": "HGNC:11985",
                        "target_canonical_id": "HGNC:672",
                        "confidence_score": 0.85,
                        "curation_status": "APPROVED",
                        "source_type": "NER",
                        "evidence_count": 1,
                        "evidence_tier": "DIRECT_LITERATURE_EVIDENCE",
                        "source_pmids": ["32132123"],
                        "source_dois": ["10.1038/s41586-020-2012-7"],
                    }

                ],
                source_label="Entity",
                target_label="Entity",
                source_key="canonical_id",
                target_key="canonical_id",
            ),
            call(
                rel_type="MENTIONS",
                edges=[
                    {
                        "source_doi": "10.1038/s41586-020-2012-7",
                        "target_canonical_id": "HGNC:11985",
                        "confidence_score": 0.9,
                    }
                ],
                source_label="Paper",
                target_label="Entity",
                source_key="doi",
                target_key="canonical_id",
            ),
        ],
        any_order=False,
    )


def test_materialize_graph_with_curation_statuses(mock_neo4j_client):
    """Verify that specifying curation_statuses correctly inserts SQL filters."""
    mock_conn = MockConnection({})
    materializer = Materializer(neo4j_client=mock_neo4j_client)

    # Run with a single status filter
    materializer.materialize_graph(pg_conn=mock_conn, curation_statuses=["APPROVED"])

    cursor = mock_conn.cursor()
    assert len(cursor.execute_calls) == 4

    # Verify relationships query has WHERE clause and params
    rel_query, rel_params = cursor.execute_calls[2]
    assert "WHERE r.curation_status = ANY(%s)" in rel_query
    assert rel_params == [["APPROVED"]]

    # Verify mentions query has WHERE clause and params
    mentions_query, mentions_params = cursor.execute_calls[3]
    assert "WHERE r.curation_status = ANY(%s)" in mentions_query
    assert mentions_params == [["APPROVED"]]


def test_materialize_graph_empty_db(mock_neo4j_client):
    """Verify materializer behaves gracefully when PostgreSQL database is empty."""
    mock_conn = MockConnection({})
    materializer = Materializer(neo4j_client=mock_neo4j_client)

    stats = materializer.materialize_graph(pg_conn=mock_conn)

    assert stats["entities"] == 0
    assert stats["papers"] == 0
    assert stats["relationships"] == 0
    assert stats["mentions"] == 0

    # Ensure Neo4j calls are skipped since there's no data
    mock_neo4j_client.bulk_upsert_nodes.assert_not_called()
    mock_neo4j_client.bulk_upsert_edges.assert_not_called()
