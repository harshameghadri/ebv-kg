"""Unit tests for Neo4jClient wrapper."""

from unittest.mock import MagicMock, call
import pytest

from app.materialization.neo4j_client import Neo4jClient


@pytest.fixture
def mock_driver():
    """Fixture providing a mock Neo4j Driver and Session."""
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    return driver, session


def test_connection_settings_defaults(monkeypatch):
    """Verify default connection settings are read from environment variables."""
    monkeypatch.setenv("NEO4J_URI", "bolt://custom-host:7687")
    monkeypatch.setenv("NEO4J_USER", "custom_user")
    monkeypatch.setenv("NEO4J_PASSWORD", "custom_pass")
    monkeypatch.setenv("NEO4J_DATABASE", "custom_db")

    mock_driver_obj = MagicMock()

    client = Neo4jClient(driver=mock_driver_obj)
    assert client.uri == "bolt://custom-host:7687"
    assert client.user == "custom_user"
    assert client.password == "custom_pass"
    assert client.database == "custom_db"


def test_connection_settings_override():
    """Verify explicit parameters override environment variables."""
    mock_driver_obj = MagicMock()

    client = Neo4jClient(
        uri="neo4j://override-host:7687",
        user="override_user",
        password="override_pass",
        database="override_db",
        driver=mock_driver_obj,
    )
    assert client.uri == "neo4j://override-host:7687"
    assert client.user == "override_user"
    assert client.password == "override_pass"
    assert client.database == "override_db"


def test_context_manager_and_close(mock_driver):
    """Verify driver.close() is called on exit or explicit close."""
    driver, _ = mock_driver

    with Neo4jClient(driver=driver) as client:
        assert client.driver == driver

    driver.close.assert_called_once()


def test_execute_query(mock_driver):
    """Verify execute_query opens session and executes Cypher statement."""
    driver, session = mock_driver
    mock_result = [MagicMock()]
    session.run.return_value = mock_result

    client = Neo4jClient(driver=driver)
    res = client.execute_query("MATCH (n) RETURN n LIMIT $limit", {"limit": 5})

    driver.session.assert_called_with(database="neo4j")
    session.run.assert_called_once_with(
        "MATCH (n) RETURN n LIMIT $limit", {"limit": 5}
    )
    assert res == mock_result


def test_execute_query_no_driver():
    """Verify RuntimeError is raised if execute_query is called with no driver."""
    client = Neo4jClient(driver=MagicMock())
    client.driver = None
    with pytest.raises(RuntimeError, match="not initialized"):
        client.execute_query("MATCH (n) RETURN n")


def test_init_schema(mock_driver):
    """Verify init_schema executes all constraint and index creation queries."""
    driver, session = mock_driver
    client = Neo4jClient(driver=driver)

    executed_queries = client.init_schema()

    assert len(executed_queries) == 5
    assert (
        "CREATE CONSTRAINT entity_canonical_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.canonical_id IS UNIQUE"
        in executed_queries
    )
    assert (
        "CREATE CONSTRAINT paper_doi_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.doi IS UNIQUE"
        in executed_queries
    )

    # Check session.run calls
    expected_calls = [call(q, {}) for q in executed_queries]
    assert session.run.call_args_list == expected_calls


def test_clear_graph(mock_driver):
    """Verify clear_graph executes DETACH DELETE query."""
    driver, session = mock_driver
    client = Neo4jClient(driver=driver)

    client.clear_graph()
    session.run.assert_called_once_with("MATCH (n) DETACH DELETE n", {})


def test_bulk_upsert_nodes_entity(mock_driver):
    """Verify bulk_upsert_nodes for Entity label defaults id_property to canonical_id."""
    driver, session = mock_driver
    client = Neo4jClient(driver=driver)

    nodes = [
        {"canonical_id": "HGNC:11985", "name": "TP53", "entity_type": "Gene"},
        {"canonical_id": "HGNC:672", "name": "BCA2", "entity_type": "Gene"},
    ]

    count = client.bulk_upsert_nodes("Entity", nodes)
    assert count == 2

    expected_query = (
        "UNWIND $nodes AS batch "
        "MERGE (n:`Entity` {`canonical_id`: batch.`canonical_id`}) "
        "SET n += batch"
    )
    session.run.assert_called_once_with(expected_query, {"nodes": nodes})


def test_bulk_upsert_nodes_paper(mock_driver):
    """Verify bulk_upsert_nodes for Paper label defaults id_property to doi."""
    driver, session = mock_driver
    client = Neo4jClient(driver=driver)

    nodes = [
        {"doi": "10.1038/s41586-020-2012-7", "title": "EBV Study 1"},
        {"doi": "10.1016/j.cell.2021.01.001", "title": "EBV Study 2"},
    ]

    count = client.bulk_upsert_nodes("Paper", nodes)
    assert count == 2

    expected_query = (
        "UNWIND $nodes AS batch "
        "MERGE (n:`Paper` {`doi`: batch.`doi`}) "
        "SET n += batch"
    )
    session.run.assert_called_once_with(expected_query, {"nodes": nodes})


def test_bulk_upsert_nodes_custom(mock_driver):
    """Verify bulk_upsert_nodes with custom label and explicit id_property."""
    driver, session = mock_driver
    client = Neo4jClient(driver=driver)

    nodes = [{"custom_id": "C123", "score": 42}]

    count = client.bulk_upsert_nodes("CustomNode", nodes, id_property="custom_id")
    assert count == 1

    expected_query = (
        "UNWIND $nodes AS batch "
        "MERGE (n:`CustomNode` {`custom_id`: batch.`custom_id`}) "
        "SET n += batch"
    )
    session.run.assert_called_once_with(expected_query, {"nodes": nodes})


def test_bulk_upsert_nodes_empty(mock_driver):
    """Verify bulk_upsert_nodes with empty nodes list returns 0 and skips driver query."""
    driver, session = mock_driver
    client = Neo4jClient(driver=driver)

    count = client.bulk_upsert_nodes("Entity", [])
    assert count == 0
    session.run.assert_not_called()


def test_bulk_upsert_nodes_cypher_injection(mock_driver):
    """Verify ValueError is raised on non-identifier label or id_property."""
    driver, _ = mock_driver
    client = Neo4jClient(driver=driver)

    with pytest.raises(ValueError, match="Invalid Cypher identifier"):
        client.bulk_upsert_nodes("Entity; DROP TABLE", [{"canonical_id": "1"}])

    with pytest.raises(ValueError, match="Invalid Cypher identifier"):
        client.bulk_upsert_nodes("Entity", [{"canonical_id": "1"}], id_property="id--")


def test_bulk_upsert_edges_basic(mock_driver):
    """Verify bulk_upsert_edges constructs parameterized Cypher and normalizes edges."""
    driver, session = mock_driver
    client = Neo4jClient(driver=driver)

    edges = [
        {
            "source_canonical_id": "HGNC:11985",
            "target_canonical_id": "HGNC:672",
            "confidence_score": 0.95,
            "curation_status": "APPROVED",
        }
    ]

    count = client.bulk_upsert_edges("INTERACTS_WITH", edges)
    assert count == 1

    expected_query = (
        "UNWIND $edges AS batch "
        "MATCH (source:`Entity` {`canonical_id`: batch.source_id}) "
        "MATCH (target:`Entity` {`canonical_id`: batch.target_id}) "
        "MERGE (source)-[r:`INTERACTS_WITH`]->(target) "
        "SET r += batch.properties"
    )
    expected_params = {
        "edges": [
            {
                "source_id": "HGNC:11985",
                "target_id": "HGNC:672",
                "properties": {
                    "confidence_score": 0.95,
                    "curation_status": "APPROVED",
                },
            }
        ]
    }
    session.run.assert_called_once_with(expected_query, expected_params)


def test_bulk_upsert_edges_paper_mentions_entity(mock_driver):
    """Verify bulk_upsert_edges between Paper and Entity with custom keys."""
    driver, session = mock_driver
    client = Neo4jClient(driver=driver)

    edges = [
        {
            "source_doi": "10.1038/s41586-020-2012-7",
            "target_canonical_id": "HGNC:11985",
            "evidence_count": 3,
        }
    ]

    count = client.bulk_upsert_edges(
        "MENTIONS",
        edges,
        source_label="Paper",
        target_label="Entity",
        source_key="doi",
        target_key="canonical_id",
    )
    assert count == 1

    expected_query = (
        "UNWIND $edges AS batch "
        "MATCH (source:`Paper` {`doi`: batch.source_id}) "
        "MATCH (target:`Entity` {`canonical_id`: batch.target_id}) "
        "MERGE (source)-[r:`MENTIONS`]->(target) "
        "SET r += batch.properties"
    )
    expected_params = {
        "edges": [
            {
                "source_id": "10.1038/s41586-020-2012-7",
                "target_id": "HGNC:11985",
                "properties": {"evidence_count": 3},
            }
        ]
    }
    session.run.assert_called_once_with(expected_query, expected_params)


def test_bulk_upsert_edges_empty(mock_driver):
    """Verify bulk_upsert_edges returns 0 and skips query when edges list is empty."""
    driver, session = mock_driver
    client = Neo4jClient(driver=driver)

    count = client.bulk_upsert_edges("TARGETS", [])
    assert count == 0
    session.run.assert_not_called()


def test_bulk_upsert_edges_missing_source_or_target(mock_driver):
    """Verify ValueError is raised if source or target ID is missing from an edge."""
    driver, _ = mock_driver
    client = Neo4jClient(driver=driver)

    with pytest.raises(ValueError, match="missing source identifier"):
        client.bulk_upsert_edges("TARGETS", [{"target_id": "B"}])

    with pytest.raises(ValueError, match="missing target identifier"):
        client.bulk_upsert_edges("TARGETS", [{"source_id": "A"}])


def test_bulk_upsert_edges_cypher_injection(mock_driver):
    """Verify ValueError is raised if rel_type or labels are invalid identifiers."""
    driver, _ = mock_driver
    client = Neo4jClient(driver=driver)

    valid_edge = [{"source_id": "A", "target_id": "B"}]

    with pytest.raises(ValueError, match="Invalid Cypher identifier"):
        client.bulk_upsert_edges("REL; DROP DATABASE", valid_edge)

    with pytest.raises(ValueError, match="Invalid Cypher identifier"):
        client.bulk_upsert_edges("REL", valid_edge, source_label="Node--")
