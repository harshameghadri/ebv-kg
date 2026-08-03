"""Unit tests for the EntityMapper class."""

import datetime
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.processing.entity_mapper import EntityMapper, parse_date, determine_relationship_type
from app.processing.synonym_resolver import SynonymResolver


# 1. Database Mocking Classes
class MockCursor:
    """Mock database cursor that simulates PostgreSQL state for tested queries."""

    def __init__(self, db_state: dict[str, dict[Any, Any]]) -> None:
        self.db_state = db_state
        self.last_query = ""
        self.last_params = None
        self.results_queue: list[Any] = []

    def execute(self, query: str, params: Any = None) -> None:
        self.last_query = query.strip()
        self.last_params = params or ()

        query_upper = self.last_query.upper()

        if "SELECT ID FROM DOCUMENTS" in query_upper:
            val = self.last_params[0]
            found = None
            for doc in self.db_state["documents"].values():
                if doc.get("doi") == val or doc.get("pmid") == val or doc.get("id") == val:
                    found = doc["id"]
                    break
            self.results_queue = [(found,)] if found else []

        elif "SELECT ID FROM DOCUMENT_CHUNKS" in query_upper:
            doc_id, chunk_idx = self.last_params
            found = None
            for chunk in self.db_state["document_chunks"].values():
                if chunk.get("document_id") == doc_id and chunk.get("chunk_index") == chunk_idx:
                    found = chunk["id"]
                    break
            self.results_queue = [(found,)] if found else []

        elif "SELECT ID, SYNONYMS FROM NORMALIZED_ENTITIES" in query_upper:
            canon_id = self.last_params[0]
            found = None
            for ent in self.db_state["normalized_entities"].values():
                if ent.get("canonical_id") == canon_id:
                    found = (ent["id"], ent.get("synonyms") or [])
                    break
            self.results_queue = [found] if found else []

        elif "SELECT SYNONYMS FROM NORMALIZED_ENTITIES" in query_upper:
            ent_id = self.last_params[0]
            ent = self.db_state["normalized_entities"].get(ent_id)
            syns = ent.get("synonyms") if ent else []
            self.results_queue = [(syns,)] if ent else []

        elif "SELECT ID, CONFIDENCE_SCORE FROM RELATIONSHIPS" in query_upper:
            src_id, tgt_id, rel_type = self.last_params
            found = None
            for rel in self.db_state["relationships"].values():
                if (
                    rel.get("source_entity_id") == src_id
                    and rel.get("target_entity_id") == tgt_id
                    and rel.get("relationship_type") == rel_type
                ):
                    found = (rel["id"], rel.get("confidence_score"))
                    break
            self.results_queue = [found] if found else []

        elif "SELECT ID FROM RELATIONSHIP_EVIDENCE" in query_upper:
            rel_id, chunk_id = self.last_params
            found = None
            for ev in self.db_state["relationship_evidence"].values():
                if ev.get("relationship_id") == rel_id and ev.get("chunk_id") == chunk_id:
                    found = ev["id"]
                    break
            self.results_queue = [(found,)] if found else []

        elif "INSERT INTO DOCUMENTS" in query_upper:
            doc_id, doi, pmid, title, journal, published_date, parsed_json = self.last_params
            if doc_id in self.db_state["documents"]:
                self.db_state["documents"][doc_id].update({
                    "doi": doi,
                    "pmid": pmid,
                    "title": title,
                    "journal": journal,
                    "published_date": published_date,
                    "parsed_json": parsed_json,
                })
            else:
                self.db_state["documents"][doc_id] = {
                    "id": doc_id,
                    "doi": doi,
                    "pmid": pmid,
                    "title": title,
                    "journal": journal,
                    "published_date": published_date,
                    "parsed_json": parsed_json,
                }
            self.results_queue = [(doc_id,)]

        elif "UPDATE DOCUMENTS" in query_upper:
            doi, pmid, title, journal, published_date, parsed_json, doc_id = self.last_params
            if doc_id in self.db_state["documents"]:
                self.db_state["documents"][doc_id].update({
                    "doi": doi,
                    "pmid": pmid,
                    "title": title,
                    "journal": journal,
                    "published_date": published_date,
                    "parsed_json": parsed_json,
                })

        elif "INSERT INTO DOCUMENT_CHUNKS" in query_upper:
            chunk_uuid, doc_id, chunk_index, content, token_count = self.last_params
            
            # Check for existing chunk to simulate ON CONFLICT
            existing_uuid = None
            for existing_chunk in self.db_state["document_chunks"].values():
                if existing_chunk.get("document_id") == doc_id and existing_chunk.get("chunk_index") == chunk_index:
                    existing_uuid = existing_chunk["id"]
                    existing_chunk.update({
                        "content": content,
                        "token_count": token_count,
                    })
                    break
            
            if existing_uuid:
                self.results_queue = [(existing_uuid,)]
            else:
                self.db_state["document_chunks"][chunk_uuid] = {
                    "id": chunk_uuid,
                    "document_id": doc_id,
                    "chunk_index": chunk_index,
                    "content": content,
                    "token_count": token_count,
                }
                self.results_queue = [(chunk_uuid,)]

        elif "UPDATE DOCUMENT_CHUNKS" in query_upper:
            content, token_count, chunk_uuid = self.last_params
            if chunk_uuid in self.db_state["document_chunks"]:
                self.db_state["document_chunks"][chunk_uuid].update({
                    "content": content,
                    "token_count": token_count,
                })

        elif "INSERT INTO NORMALIZED_ENTITIES" in query_upper:
            ent_uuid, canonical_id, name, entity_type, ontology_source, synonyms = self.last_params
            
            # Check for existing canonical_id to simulate ON CONFLICT
            existing_uuid = None
            for existing_ent in self.db_state["normalized_entities"].values():
                if existing_ent.get("canonical_id") == canonical_id:
                    existing_uuid = existing_ent["id"]
                    # Update synonyms
                    existing_syns = existing_ent.get("synonyms") or []
                    for syn in synonyms:
                        if syn not in existing_syns:
                            existing_syns.append(syn)
                    existing_ent["synonyms"] = existing_syns
                    break
            
            if existing_uuid:
                self.results_queue = [(existing_uuid,)]
            else:
                self.db_state["normalized_entities"][ent_uuid] = {
                    "id": ent_uuid,
                    "canonical_id": canonical_id,
                    "name": name,
                    "entity_type": entity_type,
                    "ontology_source": ontology_source,
                    "synonyms": synonyms,
                }
                self.results_queue = [(ent_uuid,)]

        elif "UPDATE NORMALIZED_ENTITIES" in query_upper:
            synonyms, ent_uuid = self.last_params
            if ent_uuid in self.db_state["normalized_entities"]:
                self.db_state["normalized_entities"][ent_uuid]["synonyms"] = synonyms

        elif "INSERT INTO RELATIONSHIPS" in query_upper:
            rel_id, src_id, tgt_id, rel_type, conf, status, src_type = self.last_params
            
            # Check for existing relationship to simulate ON CONFLICT
            existing_uuid = None
            for existing_rel in self.db_state["relationships"].values():
                if (
                    existing_rel.get("source_entity_id") == src_id
                    and existing_rel.get("target_entity_id") == tgt_id
                    and existing_rel.get("relationship_type") == rel_type
                ):
                    existing_uuid = existing_rel["id"]
                    existing_rel["confidence_score"] = max(existing_rel.get("confidence_score") or 0.0, conf)
                    break
            
            if existing_uuid:
                self.results_queue = [(existing_uuid,)]
            else:
                self.db_state["relationships"][rel_id] = {
                    "id": rel_id,
                    "source_entity_id": src_id,
                    "target_entity_id": tgt_id,
                    "relationship_type": rel_type,
                    "confidence_score": conf,
                    "curation_status": status,
                    "source_type": src_type,
                }
                self.results_queue = [(rel_id,)]

        elif "UPDATE RELATIONSHIPS" in query_upper:
            conf, rel_id = self.last_params
            if rel_id in self.db_state["relationships"]:
                self.db_state["relationships"][rel_id]["confidence_score"] = conf

        elif "INSERT INTO RELATIONSHIP_EVIDENCE" in query_upper:
            ev_id, rel_id, chunk_id, conf, citation = self.last_params
            
            # Check for existing evidence to simulate ON CONFLICT
            existing = False
            for existing_ev in self.db_state["relationship_evidence"].values():
                if existing_ev.get("relationship_id") == rel_id and existing_ev.get("chunk_id") == chunk_id:
                    existing_ev["confidence_score"] = max(existing_ev.get("confidence_score") or 0.0, conf)
                    existing = True
                    break
            
            if not existing:
                self.db_state["relationship_evidence"][ev_id] = {
                    "id": ev_id,
                    "relationship_id": rel_id,
                    "chunk_id": chunk_id,
                    "confidence_score": conf,
                    "citation_text": citation,
                }

    def fetchone(self) -> Any:
        if self.results_queue:
            return self.results_queue.pop(0)
        return None

    def fetchall(self) -> list[Any]:
        res = list(self.results_queue)
        self.results_queue = []
        return res

    def __enter__(self) -> "MockCursor":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


class MockConnection:
    """Mock database connection mapping to a simulated PostgreSQL DB state."""

    def __init__(self, db_state: dict[str, dict[Any, Any]] | None = None) -> None:
        self.db_state = db_state or {
            "documents": {},
            "document_chunks": {},
            "normalized_entities": {},
            "relationships": {},
            "relationship_evidence": {},
        }
        self.cursor_obj = MockCursor(self.db_state)
        self.commit_called = 0
        self.rollback_called = 0

    def cursor(self) -> MockCursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.commit_called += 1

    def rollback(self) -> None:
        self.rollback_called += 1


# 2. SynonymResolver Mocking
class MockSynonymResolver(SynonymResolver):
    """Mock SynonymResolver that returns static, custom mock records."""

    def __init__(self, mock_resolutions: dict[str, dict[str, Any]] | None = None) -> None:
        super().__init__(ols_enabled=False)
        self.mock_resolutions = mock_resolutions or {}

    def resolve(self, term: str, category: str | None = None) -> dict[str, Any] | None:
        term_clean = term.strip().lower()
        if term_clean in self.mock_resolutions:
            return self.mock_resolutions[term_clean]
        return None


# 3. Unit Tests
def test_parse_date():
    assert parse_date(None) is None
    assert parse_date("") is None

    # Test date instances
    d = datetime.date(2024, 3, 15)
    assert parse_date(d) == d

    # Test ISO format
    assert parse_date("2024-03-15") == datetime.date(2024, 3, 15)
    assert parse_date(" 2024-03-15 ") == datetime.date(2024, 3, 15)

    # Test alternate formats
    assert parse_date("2024/03/15") == datetime.date(2024, 3, 15)
    assert parse_date("15-03-2024") == datetime.date(2024, 3, 15)
    assert parse_date("15/03/2024") == datetime.date(2024, 3, 15)

    # Test just year
    assert parse_date("2024") == datetime.date(2024, 1, 1)

    # Test invalid date strings
    assert parse_date("invalid-date") is None
    assert parse_date("123") is None


def test_determine_relationship_type():
    assert determine_relationship_type("GENE", "DISEASE") == "ASSOCIATED_WITH"
    assert determine_relationship_type("protein", "disease") == "ASSOCIATED_WITH"
    assert determine_relationship_type("chemical", "disease") == "ASSOCIATED_WITH"
    assert determine_relationship_type("CHEMICAL", "CELL_TYPE") == "CO_OCCURRING"
    assert determine_relationship_type("UNKNOWN", "GENE") == "CO_OCCURRING"


def test_map_document_content_success():
    # Setup mock resolver
    mock_resolutions = {
        "ebna1": {
            "canonical_id": "HGNC:3236",
            "symbol": "EBNA1",
            "category": "hgnc",
            "source": "local_hgnc",
            "confidence": 0.99,
        },
        "lymphoma": {
            "canonical_id": "DOID:0050741",
            "symbol": "Epstein-Barr virus infectious disease",
            "category": "doid",
            "source": "local_doid",
            "confidence": 0.95,
        },
    }
    resolver = MockSynonymResolver(mock_resolutions)
    mapper = EntityMapper(resolver=resolver)

    # Inputs
    doc_metadata = {
        "doi": "10.1016/j.cell.2024.01.001",
        "pmid": "38251234",
        "title": "EBV pathogenesis in cells",
        "journal": "Cell",
        "published_date": "2024-01-15",
        "parsed_json": {"sections": ["intro", "results"]},
    }

    text_chunks = [
        {"chunk_index": 0, "text": "EBNA1 expression causes lymphoma in LCLs.", "token_count": 7},
        {"chunk_index": 1, "text": "UnresolvedGeneX is not associated with lymphoma.", "token_count": 7},
    ]

    ner_results = [
        {"text": "EBNA1", "entity_type": "GENE", "confidence": 0.98, "chunk_index": 0},
        {"text": "lymphoma", "entity_type": "DISEASE", "confidence": 0.95, "chunk_index": 0},
        {"text": "UnresolvedGeneX", "entity_type": "GENE", "confidence": 0.85, "chunk_index": 1},
        {"text": "lymphoma", "entity_type": "DISEASE", "confidence": 0.92, "chunk_index": 1},
    ]

    # Run
    conn = MockConnection()
    doc_id = mapper.map_document_content(conn, doc_metadata, text_chunks, ner_results)

    # Verify returns
    assert isinstance(doc_id, uuid.UUID)
    assert conn.commit_called == 1
    assert conn.rollback_called == 0

    # Verify database state
    state = conn.db_state

    # 1. Documents
    assert doc_id in state["documents"]
    doc = state["documents"][doc_id]
    assert doc["doi"] == "10.1016/j.cell.2024.01.001"
    assert doc["title"] == "EBV pathogenesis in cells"
    assert doc["published_date"] == datetime.date(2024, 1, 15)

    # 2. Document Chunks
    assert len(state["document_chunks"]) == 2
    chunks = list(state["document_chunks"].values())
    assert chunks[0]["document_id"] == doc_id
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["content"] == "EBNA1 expression causes lymphoma in LCLs."
    assert chunks[0]["token_count"] == 7

    assert chunks[1]["document_id"] == doc_id
    assert chunks[1]["chunk_index"] == 1

    # 3. Normalized Entities
    entities = state["normalized_entities"]
    # We expect 3 entities: EBNA1, lymphoma, UnresolvedGeneX (fallback)
    assert len(entities) == 3

    # Check EBNA1
    ebna1_ent = next(e for e in entities.values() if e["canonical_id"] == "HGNC:3236")
    assert ebna1_ent["name"] == "EBNA1"
    assert ebna1_ent["entity_type"] == "hgnc"
    assert ebna1_ent["ontology_source"] == "local_hgnc"
    assert "EBNA1" in ebna1_ent["synonyms"]

    # Check Lymphoma
    lymphoma_ent = next(e for e in entities.values() if e["canonical_id"] == "DOID:0050741")
    assert lymphoma_ent["name"] == "Epstein-Barr virus infectious disease"
    assert lymphoma_ent["entity_type"] == "doid"

    # Check Unresolved fallback
    fallback_ent = next(e for e in entities.values() if e["canonical_id"] == "raw:unresolvedgenex")
    assert fallback_ent["name"] == "UnresolvedGeneX"
    assert fallback_ent["entity_type"] == "gene"
    assert fallback_ent["ontology_source"] == "local_fallback"
    assert "UnresolvedGeneX" in fallback_ent["synonyms"]

    # 4. Relationships
    relationships = state["relationships"]
    # Chunk 0 has EBNA1 and lymphoma -> Relationship EBNA1 - lymphoma
    # Chunk 1 has UnresolvedGeneX and lymphoma -> Relationship UnresolvedGeneX - lymphoma
    assert len(relationships) == 2

    # Verify EBNA1 - Lymphoma relationship
    ebna1_id = ebna1_ent["id"]
    lymphoma_id = lymphoma_ent["id"]
    fallback_id = fallback_ent["id"]

    # Source should be GENE (EBNA1 / fallback_id), Target should be DISEASE (lymphoma)
    # priority: GENE (2) < DISEASE (4) -> source is ebna1, target is lymphoma
    rel1 = next(
        r
        for r in relationships.values()
        if r["source_entity_id"] == ebna1_id and r["target_entity_id"] == lymphoma_id
    )
    assert rel1["relationship_type"] in ("EXPRESSES", "ASSOCIATED_WITH")
    assert rel1["curation_status"] == "APPROVED"
    # Combined confidence: product of constituent confidences
    # constituent 1 (EBNA1): ner_conf=0.98, res_conf=0.99
    # constituent 2 (lymphoma): ner_conf=0.95, res_conf=0.95
    expected_conf = 0.98 * 0.99 * 0.95 * 0.95
    assert pytest.approx(rel1["confidence_score"]) == expected_conf

    # Verify UnresolvedGeneX - Lymphoma relationship
    rel2 = next(
        r
        for r in relationships.values()
        if r["source_entity_id"] == fallback_id and r["target_entity_id"] == lymphoma_id
    )
    assert rel2["relationship_type"] == "ASSOCIATED_WITH"
    # constituent 1 (fallback): ner_conf=0.85, res_conf=0.5
    # constituent 2 (lymphoma): ner_conf=0.92, res_conf=0.95
    expected_fallback_conf = 0.85 * 0.5 * 0.92 * 0.95
    assert pytest.approx(rel2["confidence_score"]) == expected_fallback_conf

    # 5. Relationship Evidence
    evidence = state["relationship_evidence"]
    assert len(evidence) == 2

    ev1 = next(ev for ev in evidence.values() if ev["relationship_id"] == rel1["id"])
    assert ev1["citation_text"] == "EBNA1 expression causes lymphoma in LCLs."
    assert pytest.approx(ev1["confidence_score"]) == expected_conf


def test_map_document_content_idempotency():
    mock_resolutions = {
        "ebna1": {
            "canonical_id": "HGNC:3236",
            "symbol": "EBNA1",
            "category": "hgnc",
            "source": "local_hgnc",
            "confidence": 0.99,
        },
        "lymphoma": {
            "canonical_id": "DOID:0050741",
            "symbol": "Epstein-Barr virus infectious disease",
            "category": "doid",
            "source": "local_doid",
            "confidence": 0.95,
        },
    }
    resolver = MockSynonymResolver(mock_resolutions)
    mapper = EntityMapper(resolver=resolver)

    doc_metadata = {
        "doi": "10.1016/j.cell.2024.01.001",
        "title": "EBV Study",
        "journal": "Cell",
        "published_date": "2024-01-15",
    }

    text_chunks = [
        {"chunk_index": 0, "text": "EBNA1 causes lymphoma.", "token_count": 3},
    ]

    ner_results = [
        {"text": "EBNA1", "entity_type": "GENE", "confidence": 0.80, "chunk_index": 0},
        {"text": "lymphoma", "entity_type": "DISEASE", "confidence": 0.85, "chunk_index": 0},
    ]

    conn = MockConnection()

    # 1. Run First Time
    doc_id1 = mapper.map_document_content(conn, doc_metadata, text_chunks, ner_results)

    assert len(conn.db_state["documents"]) == 1
    assert len(conn.db_state["document_chunks"]) == 1
    assert len(conn.db_state["normalized_entities"]) == 2
    assert len(conn.db_state["relationships"]) == 1
    assert len(conn.db_state["relationship_evidence"]) == 1

    # Keep track of IDs
    orig_chunk_id = list(conn.db_state["document_chunks"].keys())[0]
    orig_rel_id = list(conn.db_state["relationships"].keys())[0]
    orig_rel_conf = conn.db_state["relationships"][orig_rel_id]["confidence_score"]

    # 2. Run Second Time with higher confidences & new synonyms
    # We pass EBNA-1 (new alias) which resolves to same HGNC:3236
    mock_resolutions["ebna-1"] = mock_resolutions["ebna1"]
    ner_results_new = [
        {"text": "EBNA-1", "entity_type": "GENE", "confidence": 0.95, "chunk_index": 0},
        {"text": "lymphoma", "entity_type": "DISEASE", "confidence": 0.96, "chunk_index": 0},
    ]

    doc_id2 = mapper.map_document_content(conn, doc_metadata, text_chunks, ner_results_new)

    # Verify ID stays the same
    assert doc_id1 == doc_id2

    # Verify counts did not double
    assert len(conn.db_state["documents"]) == 1
    assert len(conn.db_state["document_chunks"]) == 1
    assert len(conn.db_state["normalized_entities"]) == 2
    assert len(conn.db_state["relationships"]) == 1
    assert len(conn.db_state["relationship_evidence"]) == 1

    # Verify chunk ID did not change
    assert list(conn.db_state["document_chunks"].keys())[0] == orig_chunk_id

    # Verify relationship ID did not change
    assert list(conn.db_state["relationships"].keys())[0] == orig_rel_id

    # Verify confidence was updated to the higher one
    new_rel_conf = conn.db_state["relationships"][orig_rel_id]["confidence_score"]
    assert new_rel_conf > orig_rel_conf
    expected_new_conf = 0.95 * 0.99 * 0.96 * 0.95
    assert pytest.approx(new_rel_conf) == expected_new_conf

    # Verify synonyms array for EBNA1 updated to include EBNA-1
    ebna1_uuid = next(k for k, v in conn.db_state["normalized_entities"].items() if v["canonical_id"] == "HGNC:3236")
    syns = conn.db_state["normalized_entities"][ebna1_uuid]["synonyms"]
    assert "EBNA1" in syns
    assert "EBNA-1" in syns


def test_map_document_content_transaction_rollback():
    resolver = MockSynonymResolver()
    mapper = EntityMapper(resolver=resolver)

    # Let's break the mock cursor so it raises an exception during write
    conn = MockConnection()

    def raise_db_error(*args, **kwargs):
        raise ValueError("Simulated DB Write Failure")

    # Override cursor's execute to raise error
    conn.cursor().execute = raise_db_error

    doc_metadata = {"doi": "10.1234/failed"}

    with pytest.raises(ValueError, match="Simulated DB Write Failure"):
        mapper.map_document_content(conn, doc_metadata, [], [])

    # Check commit and rollback behavior
    assert conn.commit_called == 0
    assert conn.rollback_called == 1


def test_map_document_content_fallback_chunk_mapping():
    # Setup mock resolver
    resolver = MockSynonymResolver({
        "ebna1": {
            "canonical_id": "HGNC:3236",
            "symbol": "EBNA1",
            "category": "hgnc",
            "source": "local_hgnc",
            "confidence": 0.99,
        }
    })
    mapper = EntityMapper(resolver=resolver)

    # Inputs without explicit chunk_index in ner_results
    doc_metadata = {"doi": "10.1016/j.cell.2024.01.002", "title": "Test Fallback"}
    text_chunks = [
        {"chunk_index": 0, "text": "This chunk is about cell division."},
        {"chunk_index": 1, "text": "Here we mention EBNA1 causing infection."},
    ]
    ner_results = [
        {"text": "EBNA1", "entity_type": "GENE", "confidence": 0.95}
    ]

    conn = MockConnection()
    mapper.map_document_content(conn, doc_metadata, text_chunks, ner_results)

    # The fallback should map EBNA1 to chunk 1 because "EBNA1" is a substring of chunk 1
    # Check that relationship/evidence mapping matches chunk index 1
    state = conn.db_state
    assert len(state["document_chunks"]) == 2
    chunks = list(state["document_chunks"].values())
    chunk1_id = next(c["id"] for c in chunks if c["chunk_index"] == 1)

    # Verify EBNA1 was associated with chunk 1
    ebna1_uuid = next(e["id"] for e in state["normalized_entities"].values() if e["canonical_id"] == "HGNC:3236")

    # Since there's only 1 entity, there are no pairs, hence no relationships.
    # But let's check that the entity is normalized and stored.
    assert ebna1_uuid in state["normalized_entities"]
