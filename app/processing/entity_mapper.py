"""Entity Mapping module for processing raw NER entities and storing them in PostgreSQL."""

import datetime
import json
import logging
import uuid
import itertools
from collections import defaultdict
from typing import Any

from app.processing.synonym_resolver import SynonymResolver, normalize_term

logger = logging.getLogger(__name__)

# Priorities for determining relationship direction. Lower priority goes first as source_entity_id.
TYPE_PRIORITY: dict[str, int] = {
    "CHEMICAL": 1,
    "GENE": 2,
    "PROTEIN": 2,
    "CELL_TYPE": 3,
    "DISEASE": 4,
}


def parse_date(date_val: Any) -> Any:
    """Helper to parse a date robustly into a datetime.date object or None."""
    if not date_val:
        return None
    if isinstance(date_val, datetime.date):
        return date_val
    if isinstance(date_val, str):
        date_str = date_val.strip()
        # Try ISO format (YYYY-MM-DD)
        try:
            return datetime.date.fromisoformat(date_str)
        except ValueError:
            pass
        # Try various common formats
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
            try:
                return datetime.datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
        # Try just YYYY
        if len(date_str) == 4 and date_str.isdigit():
            try:
                return datetime.date(int(date_str), 1, 1)
            except ValueError:
                pass
    return None


def determine_relationship_type(t1: str, t2: str) -> str:
    """Determine a logical relationship type based on the entity types."""
    types = {t1.upper(), t2.upper()}
    if "GENE" in types and "DISEASE" in types:
        return "ASSOCIATED_WITH"
    if "PROTEIN" in types and "DISEASE" in types:
        return "ASSOCIATED_WITH"
    if "CHEMICAL" in types and "DISEASE" in types:
        return "ASSOCIATED_WITH"  # or TREATS. Standardize to ASSOCIATED_WITH per spec details
    return "CO_OCCURRING"


class EntityMapper:
    """Processes raw NER entities, normalizes them, and updates PostgreSQL database."""

    def __init__(self, resolver: SynonymResolver | None = None) -> None:
        """Initialize EntityMapper with a SynonymResolver."""
        self.resolver = resolver or SynonymResolver()

    def map_document_content(
        self,
        conn: Any,
        doc_metadata: dict[str, Any],
        text_chunks: list[dict[str, Any]],
        ner_results: list[dict[str, Any]],
    ) -> uuid.UUID:
        """Map document metadata, chunks, raw entities, and relationships to PostgreSQL.

        Runs within a single transaction. Commits at the end or rolls back on error.

        Args:
            conn: A PostgreSQL connection object (psycopg-compatible).
            doc_metadata: Metadata for the document (doi, pmid, title, journal, etc.).
            text_chunks: List of document chunks (each should have content/text and optionally chunk_index).
            ner_results: List of raw extracted entities from the document.

        Returns:
            The UUID of the processed document.
        """
        try:
            with conn.cursor() as cursor:
                # 1. Resolve Document ID and Upsert/Insert Document Metadata
                doc_id = doc_metadata.get("id")
                if doc_id:
                    if isinstance(doc_id, str):
                        doc_id = uuid.UUID(doc_id)
                else:
                    # Look up existing document by DOI or PMID
                    doi = doc_metadata.get("doi")
                    pmid = doc_metadata.get("pmid")
                    if doi:
                        cursor.execute("SELECT id FROM documents WHERE doi = %s", (doi,))
                        row = cursor.fetchone()
                        if row:
                            doc_id = row[0]
                    if not doc_id and pmid:
                        cursor.execute("SELECT id FROM documents WHERE pmid = %s", (pmid,))
                        row = cursor.fetchone()
                        if row:
                            doc_id = row[0]
                    # Generate a new ID if not found
                    if not doc_id:
                        doc_id = uuid.uuid4()

                # Upsert Document
                doi = doc_metadata.get("doi")
                pmid = doc_metadata.get("pmid")
                title = doc_metadata.get("title")
                journal = doc_metadata.get("journal")
                pub_date = parse_date(doc_metadata.get("published_date") or doc_metadata.get("publication_date"))
                parsed_json = doc_metadata.get("parsed_json")
                parsed_json_str = json.dumps(parsed_json) if parsed_json is not None else None

                # Query if exists
                cursor.execute("SELECT id FROM documents WHERE id = %s", (doc_id,))
                if cursor.fetchone():
                    cursor.execute(
                        """
                        UPDATE documents
                        SET doi = %s, pmid = %s, title = %s, journal = %s, published_date = %s, parsed_json = %s
                        WHERE id = %s
                        """,
                        (doi, pmid, title, journal, pub_date, parsed_json_str, doc_id)
                    )
                else:
                    cursor.execute(
                        """
                        INSERT INTO documents (id, doi, pmid, title, journal, published_date, parsed_json)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (doc_id, doi, pmid, title, journal, pub_date, parsed_json_str)
                    )

                # 2. Insert/Upsert Text Chunks
                inserted_chunks = []
                for idx, chunk in enumerate(text_chunks):
                    # Check for chunk_index, fallback to loop index
                    chunk_index = chunk.get("chunk_index") or chunk.get("index")
                    if chunk_index is None:
                        chunk_index = idx
                    else:
                        chunk_index = int(chunk_index)

                    content = chunk.get("content") or chunk.get("text") or ""
                    token_count = chunk.get("token_count") or len(content.split())

                    # Check if chunk already exists for this document and index
                    cursor.execute(
                        "SELECT id FROM document_chunks WHERE document_id = %s AND chunk_index = %s",
                        (doc_id, chunk_index)
                    )
                    row = cursor.fetchone()
                    if row:
                        chunk_uuid = row[0]
                        cursor.execute(
                            "UPDATE document_chunks SET content = %s, token_count = %s WHERE id = %s",
                            (content, token_count, chunk_uuid)
                        )
                    else:
                        chunk_uuid = uuid.uuid4()
                        cursor.execute(
                            """
                            INSERT INTO document_chunks (id, document_id, chunk_index, content, token_count)
                            VALUES (%s, %s, %s, %s, %s)
                            """,
                            (chunk_uuid, doc_id, chunk_index, content, token_count)
                        )

                    inserted_chunks.append({
                        "uuid": chunk_uuid,
                        "chunk_index": chunk_index,
                        "content": content,
                    })

                # 3. Resolve and Normalize Entities
                # We will keep an in-memory mapping of canonical_id -> entity_uuid to speed up inserts
                canonical_id_to_uuid = {}
                chunk_entities = defaultdict(list)

                for raw_ent in ner_results:
                    entity_text = raw_ent.get("text") or raw_ent.get("mention") or raw_ent.get("name") or ""
                    entity_text = entity_text.strip()
                    if not entity_text:
                        continue

                    entity_type = raw_ent.get("entity_type") or raw_ent.get("category") or raw_ent.get("type") or "unknown"
                    ner_confidence = float(raw_ent.get("confidence") or raw_ent.get("prob") or raw_ent.get("score") or 1.0)

                    # Resolve using SynonymResolver
                    res = self.resolver.resolve(entity_text, category=entity_type)
                    if res:
                        canonical_id = res["canonical_id"]
                        symbol = res["symbol"]
                        category = res["category"]
                        ontology_source = res["source"]
                        resolution_confidence = float(res["confidence"])
                    else:
                        normalized_name = normalize_term(entity_text).replace(" ", "_")
                        canonical_id = f"raw:{normalized_name}"
                        symbol = entity_text
                        category = entity_type.lower()
                        ontology_source = "local_fallback"
                        resolution_confidence = 0.5

                    # Write / Upsert Normalized Entity
                    if canonical_id in canonical_id_to_uuid:
                        entity_uuid = canonical_id_to_uuid[canonical_id]
                        # Fetch and update synonyms if needed
                        cursor.execute("SELECT synonyms FROM normalized_entities WHERE id = %s", (entity_uuid,))
                        syn_row = cursor.fetchone()
                        if syn_row:
                            synonyms = syn_row[0] or []
                            if entity_text not in synonyms:
                                synonyms.append(entity_text)
                                cursor.execute(
                                    "UPDATE normalized_entities SET synonyms = %s WHERE id = %s",
                                    (synonyms, entity_uuid)
                                )
                    else:
                        # Query DB to see if canonical_id exists
                        cursor.execute(
                            "SELECT id, synonyms FROM normalized_entities WHERE canonical_id = %s",
                            (canonical_id,)
                        )
                        row = cursor.fetchone()
                        if row:
                            entity_uuid = row[0]
                            canonical_id_to_uuid[canonical_id] = entity_uuid
                            synonyms = row[1] or []
                            if entity_text not in synonyms:
                                synonyms.append(entity_text)
                                cursor.execute(
                                    "UPDATE normalized_entities SET synonyms = %s WHERE id = %s",
                                    (synonyms, entity_uuid)
                                )
                        else:
                            entity_uuid = uuid.uuid4()
                            canonical_id_to_uuid[canonical_id] = entity_uuid
                            cursor.execute(
                                """
                                INSERT INTO normalized_entities (id, canonical_id, name, entity_type, ontology_source, synonyms)
                                VALUES (%s, %s, %s, %s, %s, %s)
                                """,
                                (
                                    entity_uuid,
                                    canonical_id,
                                    symbol,
                                    category,
                                    ontology_source,
                                    [entity_text],
                                )
                            )

                    # Map Entity to Chunk
                    # Determine which chunk it belongs to
                    mapped_chunk = None
                    chunk_index_val = raw_ent.get("chunk_index")
                    if chunk_index_val is not None:
                        chunk_index_int = int(chunk_index_val)
                        # Find the inserted chunk with matching chunk_index
                        for c in inserted_chunks:
                            if c["chunk_index"] == chunk_index_int:
                                mapped_chunk = c
                                break
                    elif raw_ent.get("chunk_id") is not None:
                        chunk_id_val = uuid.UUID(str(raw_ent["chunk_id"]))
                        for c in inserted_chunks:
                            if c["uuid"] == chunk_id_val:
                                mapped_chunk = c
                                break

                    # Fallback: substring matching in chunks
                    if not mapped_chunk and inserted_chunks:
                        for c in inserted_chunks:
                            if entity_text.lower() in c["content"].lower():
                                mapped_chunk = c
                                break
                        # If still not mapped, default to first chunk
                        if not mapped_chunk:
                            mapped_chunk = inserted_chunks[0]

                    if mapped_chunk:
                        chunk_entities[mapped_chunk["uuid"]].append({
                            "uuid": entity_uuid,
                            "canonical_id": canonical_id,
                            "entity_type": entity_type,
                            "ner_confidence": ner_confidence,
                            "resolution_confidence": resolution_confidence,
                        })

                # 4. Insert/Upsert Relationships and Evidence
                for chunk_uuid, entities in chunk_entities.items():
                    # Find chunk content
                    chunk_content = ""
                    for c in inserted_chunks:
                        if c["uuid"] == chunk_uuid:
                            chunk_content = c["content"]
                            break

                    # Deduplicate entities by canonical_id within the same chunk
                    # keeping the one with higher combined confidence if duplicates exist
                    unique_chunk_entities = {}
                    for ent in entities:
                        cid = ent["canonical_id"]
                        if cid not in unique_chunk_entities:
                            unique_chunk_entities[cid] = ent
                        else:
                            # Keep highest product of confidences
                            curr_conf = ent["ner_confidence"] * ent["resolution_confidence"]
                            prev_conf = (
                                unique_chunk_entities[cid]["ner_confidence"]
                                * unique_chunk_entities[cid]["resolution_confidence"]
                            )
                            if curr_conf > prev_conf:
                                unique_chunk_entities[cid] = ent

                    unique_list = list(unique_chunk_entities.values())

                    # Insert relationship for all pairs of resolved entities
                    for ent1, ent2 in itertools.combinations(unique_list, 2):
                        # Determine relationship direction based on TYPE_PRIORITY
                        p1 = TYPE_PRIORITY.get(ent1["entity_type"].upper(), 5)
                        p2 = TYPE_PRIORITY.get(ent2["entity_type"].upper(), 5)

                        if p1 < p2:
                            source_ent, target_ent = ent1, ent2
                        elif p1 > p2:
                            source_ent, target_ent = ent2, ent1
                        else:
                            if ent1["canonical_id"] < ent2["canonical_id"]:
                                source_ent, target_ent = ent1, ent2
                            else:
                                source_ent, target_ent = ent2, ent1

                        source_entity_id = source_ent["uuid"]
                        target_entity_id = target_ent["uuid"]

                        rel_type = determine_relationship_type(
                            source_ent["entity_type"], target_ent["entity_type"]
                        )

                        # Combined confidence: product of the constituent NER/resolution confidences
                        combined_confidence = (
                            source_ent["ner_confidence"]
                            * source_ent["resolution_confidence"]
                            * target_ent["ner_confidence"]
                            * target_ent["resolution_confidence"]
                        )

                        # Upsert relationship
                        cursor.execute(
                            """
                            SELECT id, confidence_score FROM relationships
                            WHERE source_entity_id = %s AND target_entity_id = %s AND relationship_type = %s
                            """,
                            (source_entity_id, target_entity_id, rel_type)
                        )
                        rel_row = cursor.fetchone()
                        if rel_row:
                            relationship_id = rel_row[0]
                            existing_conf = rel_row[1] or 0.0
                            if combined_confidence > existing_conf:
                                cursor.execute(
                                    "UPDATE relationships SET confidence_score = %s WHERE id = %s",
                                    (combined_confidence, relationship_id)
                                )
                        else:
                            relationship_id = uuid.uuid4()
                            cursor.execute(
                                """
                                INSERT INTO relationships (id, source_entity_id, target_entity_id, relationship_type, confidence_score, curation_status, source_type)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                                """,
                                (
                                    relationship_id,
                                    source_entity_id,
                                    target_entity_id,
                                    rel_type,
                                    combined_confidence,
                                    "PENDING",
                                    "text_mining"
                                )
                            )

                        # Insert Relationship Evidence
                        cursor.execute(
                            "SELECT id FROM relationship_evidence WHERE relationship_id = %s AND chunk_id = %s",
                            (relationship_id, chunk_uuid)
                        )
                        ev_row = cursor.fetchone()
                        if not ev_row:
                            cursor.execute(
                                """
                                INSERT INTO relationship_evidence (id, relationship_id, chunk_id, confidence_score, citation_text)
                                VALUES (%s, %s, %s, %s, %s)
                                """,
                                (
                                    uuid.uuid4(),
                                    relationship_id,
                                    chunk_uuid,
                                    combined_confidence,
                                    chunk_content
                                )
                            )

            # If all operations succeed, commit the transaction
            conn.commit()
            return doc_id

        except Exception as e:
            # Rollback transaction on error
            try:
                conn.rollback()
            except Exception as rollback_err:
                logger.error("Database rollback failed: %s", rollback_err)
            logger.error("Error mapping document content: %s", e)
            raise e
