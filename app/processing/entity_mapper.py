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


def determine_relationship_type(t1: str, t2: str, chunk_content: str = "") -> str:
    """Determine a logical relationship type based on entity types and context text."""
    t1_u, t2_u = t1.upper(), t2.upper()
    types = {t1_u, t2_u}
    content_lower = chunk_content.lower()

    if "express" in content_lower or "induce" in content_lower or "upregulat" in content_lower:
        return "EXPRESSES"
    elif "inhib" in content_lower or "block" in content_lower or "suppress" in content_lower:
        return "INHIBITS"
    elif "bind" in content_lower or "interact" in content_lower:
        return "INTERACTS_WITH"
    elif "GENE" in types and "DISEASE" in types:
        return "ASSOCIATED_WITH"
    elif "PROTEIN" in types and "DISEASE" in types:
        return "ASSOCIATED_WITH"
    elif "CHEMICAL" in types and "DISEASE" in types:
        return "TREATS" if ("treat" in content_lower or "therap" in content_lower or "drug" in content_lower) else "ASSOCIATED_WITH"
    return "CO_OCCURRING"
    
# priorities continue below...
TYPE_PRIORITY: dict[str, int] = {
    "CHEMICAL": 1,
    "GENE": 2,
    "PROTEIN": 2,
    "CELL_TYPE": 3,
    "DISEASE": 4,
}


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
                # 1. Resolve Document ID and Upsert/Insert Document Metadata concurrently-safely
                doc_id = doc_metadata.get("id")
                doi = doc_metadata.get("doi")
                pmid = doc_metadata.get("pmid")

                if doc_id:
                    if isinstance(doc_id, str):
                        doc_id = uuid.UUID(doc_id)
                else:
                    # Check if document already exists by DOI or PMID
                    existing_id = None
                    if doi:
                        cursor.execute("SELECT id FROM documents WHERE doi = %s", (doi,))
                        row = cursor.fetchone()
                        if row:
                            existing_id = row[0]
                    if not existing_id and pmid:
                        cursor.execute("SELECT id FROM documents WHERE pmid = %s", (pmid,))
                        row = cursor.fetchone()
                        if row:
                            existing_id = row[0]

                    if existing_id:
                        doc_id = existing_id
                    elif doi:
                        doc_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"doi:{doi}")
                    elif pmid:
                        doc_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"pmid:{pmid}")
                    else:
                        doc_id = uuid.uuid4()

                title = doc_metadata.get("title")
                journal = doc_metadata.get("journal")
                pub_date = parse_date(doc_metadata.get("published_date") or doc_metadata.get("publication_date"))
                parsed_json = doc_metadata.get("parsed_json")
                parsed_json_str = json.dumps(parsed_json) if parsed_json is not None else None

                # Concurrent-safe atomic document upsert
                cursor.execute(
                    """
                    INSERT INTO documents (id, doi, pmid, title, journal, published_date, parsed_json)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE 
                    SET doi = EXCLUDED.doi, pmid = EXCLUDED.pmid, title = EXCLUDED.title, 
                        journal = EXCLUDED.journal, published_date = EXCLUDED.published_date, 
                        parsed_json = EXCLUDED.parsed_json
                    RETURNING id;
                    """,
                    (doc_id, doi, pmid, title, journal, pub_date, parsed_json_str)
                )
                row = cursor.fetchone()
                if row:
                    doc_id = row[0]

                # 2. Insert/Upsert Text Chunks
                inserted_chunks = []
                for idx, chunk in enumerate(text_chunks):
                    chunk_index = chunk.get("chunk_index") or chunk.get("index")
                    if chunk_index is None:
                        chunk_index = idx
                    else:
                        chunk_index = int(chunk_index)

                    content = chunk.get("content") or chunk.get("text") or ""
                    token_count = chunk.get("token_count") or len(content.split())

                    # Concurrent-safe document chunks upsert
                    chunk_uuid = uuid.uuid4()
                    cursor.execute(
                        """
                        INSERT INTO document_chunks (id, document_id, chunk_index, content, token_count)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (document_id, chunk_index) DO UPDATE 
                        SET content = EXCLUDED.content, token_count = EXCLUDED.token_count
                        RETURNING id;
                        """,
                        (chunk_uuid, doc_id, chunk_index, content, token_count)
                    )
                    row = cursor.fetchone()
                    if row:
                        chunk_uuid = row[0]

                    inserted_chunks.append({
                        "uuid": chunk_uuid,
                        "chunk_index": chunk_index,
                        "content": content,
                    })

                # 3. Resolve, Group, and Sort Entities to prevent database write locks deadlocks
                resolved_ents = []
                for raw_ent in ner_results:
                    entity_text = raw_ent.get("text") or raw_ent.get("mention") or raw_ent.get("name") or ""
                    entity_text = entity_text.strip()
                    if not entity_text:
                        continue

                    entity_type = raw_ent.get("entity_type") or raw_ent.get("category") or raw_ent.get("type") or "unknown"
                    ner_confidence = float(raw_ent.get("confidence") or raw_ent.get("prob") or raw_ent.get("score") or 1.0)

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

                    resolved_ents.append({
                        "raw_ent": raw_ent,
                        "canonical_id": canonical_id,
                        "symbol": symbol,
                        "category": category,
                        "ontology_source": ontology_source,
                        "resolution_confidence": resolution_confidence,
                        "entity_text": entity_text,
                        "entity_type": entity_type,
                        "ner_confidence": ner_confidence,
                    })

                # Sort by canonical_id lexicographically so all parallel tasks lock entities in the same deterministic order
                resolved_ents.sort(key=lambda x: x["canonical_id"])

                canonical_id_to_uuid = {}
                chunk_entities = defaultdict(list)

                for ent in resolved_ents:
                    canonical_id = ent["canonical_id"]
                    entity_text = ent["entity_text"]
                    symbol = ent["symbol"]
                    category = ent["category"]
                    ontology_source = ent["ontology_source"]
                    entity_type = ent["entity_type"]
                    ner_confidence = ent["ner_confidence"]
                    resolution_confidence = ent["resolution_confidence"]
                    raw_ent = ent["raw_ent"]

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
                        entity_uuid = uuid.uuid4()
                        cursor.execute(
                            """
                            INSERT INTO normalized_entities (id, canonical_id, name, entity_type, ontology_source, synonyms)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (canonical_id) DO UPDATE 
                            SET synonyms = ARRAY(SELECT DISTINCT unnest(array_cat(COALESCE(normalized_entities.synonyms, '{}'), EXCLUDED.synonyms)))
                            RETURNING id;
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
                        row = cursor.fetchone()
                        if row:
                            entity_uuid = row[0]
                        canonical_id_to_uuid[canonical_id] = entity_uuid

                    # Map Entity to Chunk
                    mapped_chunk = None
                    chunk_index_val = raw_ent.get("chunk_index")
                    if chunk_index_val is not None:
                        chunk_index_int = int(chunk_index_val)
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

                    if not mapped_chunk and inserted_chunks:
                        for c in inserted_chunks:
                            if entity_text.lower() in c["content"].lower():
                                mapped_chunk = c
                                break
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

                # 4. Collect and Deduplicate Relationships & Evidence across all chunks in memory
                relationships_to_upsert = {}

                for chunk_uuid, entities in chunk_entities.items():
                    chunk_content = ""
                    for c in inserted_chunks:
                        if c["uuid"] == chunk_uuid:
                            chunk_content = c["content"]
                            break

                    unique_chunk_entities = {}
                    for ent in entities:
                        cid = ent["canonical_id"]
                        if cid not in unique_chunk_entities:
                            unique_chunk_entities[cid] = ent
                        else:
                            curr_conf = ent["ner_confidence"] * ent["resolution_confidence"]
                            prev_conf = (
                                unique_chunk_entities[cid]["ner_confidence"]
                                * unique_chunk_entities[cid]["resolution_confidence"]
                            )
                            if curr_conf > prev_conf:
                                unique_chunk_entities[cid] = ent

                    unique_list = list(unique_chunk_entities.values())

                    for ent1, ent2 in itertools.combinations(unique_list, 2):
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
                            source_ent["entity_type"], target_ent["entity_type"], chunk_content
                        )

                        combined_confidence = (
                            source_ent["ner_confidence"]
                            * source_ent["resolution_confidence"]
                            * target_ent["ner_confidence"]
                            * target_ent["resolution_confidence"]
                        )

                        rel_key = (source_entity_id, target_entity_id, rel_type)
                        
                        if rel_key not in relationships_to_upsert:
                            relationships_to_upsert[rel_key] = {
                                "source_entity_id": source_entity_id,
                                "target_entity_id": target_entity_id,
                                "rel_type": rel_type,
                                "combined_confidence": combined_confidence,
                                "evidence": []
                            }
                        else:
                            if combined_confidence > relationships_to_upsert[rel_key]["combined_confidence"]:
                                relationships_to_upsert[rel_key]["combined_confidence"] = combined_confidence
                        
                        relationships_to_upsert[rel_key]["evidence"].append({
                            "chunk_uuid": chunk_uuid,
                            "confidence_score": combined_confidence,
                            "citation_text": chunk_content
                        })

                # Sort relationships lexicographically to prevent database deadlocks on concurrent writes
                sorted_rel_keys = sorted(relationships_to_upsert.keys(), key=lambda k: (str(k[0]), str(k[1]), k[2]))

                # 5. Write/Upsert Relationships & Evidence in sorted order
                for rel_key in sorted_rel_keys:
                    rel_data = relationships_to_upsert[rel_key]
                    source_entity_id = rel_data["source_entity_id"]
                    target_entity_id = rel_data["target_entity_id"]
                    rel_type = rel_data["rel_type"]
                    combined_confidence = rel_data["combined_confidence"]

                    # Determine 3-tier curation status
                    if combined_confidence >= 0.80:
                        curation_status = "APPROVED"
                    else:
                        curation_status = "PENDING"

                    relationship_id = uuid.uuid4()
                    cursor.execute(
                        """
                        INSERT INTO relationships (id, source_entity_id, target_entity_id, relationship_type, confidence_score, curation_status, source_type)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (source_entity_id, target_entity_id, relationship_type) DO UPDATE 
                        SET confidence_score = GREATEST(relationships.confidence_score, EXCLUDED.confidence_score)
                        RETURNING id;
                        """,
                        (
                            relationship_id,
                            source_entity_id,
                            target_entity_id,
                            rel_type,
                            combined_confidence,
                            curation_status,
                            "text_mining"
                        )
                    )
                    row = cursor.fetchone()
                    if row:
                        relationship_id = row[0]

                    # Insert evidence (also sorted by chunk_uuid to prevent lock ordering conflicts)
                    sorted_evidence = sorted(rel_data["evidence"], key=lambda e: str(e["chunk_uuid"]))
                    for ev in sorted_evidence:
                        cursor.execute(
                            """
                            INSERT INTO relationship_evidence (id, relationship_id, chunk_id, confidence_score, citation_text)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (relationship_id, chunk_id) DO UPDATE 
                            SET confidence_score = GREATEST(relationship_evidence.confidence_score, EXCLUDED.confidence_score);
                            """,
                            (
                                uuid.uuid4(),
                                relationship_id,
                                ev["chunk_uuid"],
                                ev["confidence_score"],
                                ev["citation_text"]
                            )
                        )

            conn.commit()
            return doc_id

        except Exception as e:
            try:
                conn.rollback()
            except Exception as rollback_err:
                logger.error("Database rollback failed: %s", rollback_err)
            logger.error("Error mapping document content: %s", e)
            raise e
