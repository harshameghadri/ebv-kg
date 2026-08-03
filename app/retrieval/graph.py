"""Graph-augmented retrieval module utilizing Neo4jClient to fetch multi-hop neighborhood context."""

import logging
import re
from typing import Any, Dict, List, Optional, Set

from app.materialization.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class GraphRetriever:
    """Retrieves multi-hop entity and paper neighborhood context from Neo4j."""

    def __init__(
        self,
        neo4j_client: Optional[Neo4jClient] = None,
        min_confidence: float = 0.70,
        synonym_resolver: Optional[Any] = None,
        ner_extractor: Optional[Any] = None,
    ) -> None:
        """Initialize GraphRetriever.

        Args:
            neo4j_client: An instance of Neo4jClient. If None, a new one is created.
            min_confidence: Minimum confidence score threshold for relationship traversal.
            synonym_resolver: Optional synonym resolver instance.
            ner_extractor: Optional NER extractor instance.
        """
        self.neo4j_client = neo4j_client or Neo4jClient()
        self.min_confidence = min_confidence
        self.synonym_resolver = synonym_resolver
        self.ner_extractor = ner_extractor

    def _find_entities_by_name(self, term: str) -> List[str]:
        """Query Neo4j for entities matching name or synonyms.

        Args:
            term: Entity term to search.

        Returns:
            List of canonical IDs matching the term.
        """
        cypher = """
        MATCH (e:Entity)
        WHERE toLower(e.name) = toLower($term)
           OR any(syn in e.synonyms WHERE toLower(syn) = toLower($term))
        RETURN e.canonical_id AS canonical_id
        """
        try:
            res = self.neo4j_client.execute_query(cypher, {"term": term})
            ids = []
            for r in res:
                # Handle different neo4j driver result formats (Record vs dict)
                try:
                    cid = r.get("canonical_id")
                except AttributeError:
                    cid = dict(r).get("canonical_id")
                if cid:
                    ids.append(cid)
            return ids
        except Exception as e:
            logger.warning("Error finding entities by name '%s': %s", term, e)
            return []

    def extract_candidates(self, query: str) -> List[str]:
        """Extract candidate entity canonical IDs from a query string.

        First attempts NER & SynonymResolver if configured. Falls back to
        a simple word-boundary keyword search against all Entity nodes in Neo4j.

        Args:
            query: User search query.

        Returns:
            List of canonical IDs of extracted entities.
        """
        if not query or not query.strip():
            return []

        canonical_ids: Set[str] = set()

        # 1. Try NER extractor if available
        if self.ner_extractor:
            try:
                extracted = self.ner_extractor.extract(query)
                for ent in extracted:
                    term = ent.get("text")
                    category = ent.get("entity_type")
                    if term:
                        if self.synonym_resolver:
                            resolved = self.synonym_resolver.resolve(term, category=category)
                            if resolved and resolved.get("canonical_id"):
                                canonical_ids.add(resolved["canonical_id"])
                        else:
                            # Direct name matching in Neo4j
                            matches = self._find_entities_by_name(term)
                            for m in matches:
                                canonical_ids.add(m)
            except Exception as e:
                logger.warning("Error during NER candidate extraction: %s", e)

        # 2. Keyword search against all known Entity nodes in Neo4j
        if not canonical_ids:
            try:
                cypher = """
                MATCH (e:Entity)
                RETURN e.canonical_id AS canonical_id, e.name AS name, e.synonyms AS synonyms
                """
                results = self.neo4j_client.execute_query(cypher)
                lowered_query = query.lower()

                for record in results:
                    try:
                        canonical_id = record.get("canonical_id")
                        name = record.get("name")
                        synonyms = record.get("synonyms") or []
                    except (AttributeError, KeyError):
                        rec_dict = dict(record)
                        canonical_id = rec_dict.get("canonical_id")
                        name = rec_dict.get("name")
                        synonyms = rec_dict.get("synonyms") or []

                    if not canonical_id or not name:
                        continue

                    # Check for exact word or phrase match of name or synonyms
                    terms_to_check = [name] + list(synonyms)
                    matched = False
                    for term in terms_to_check:
                        if not term:
                            continue
                        escaped = re.escape(term.strip().lower())
                        pattern = rf"\b{escaped}\b"
                        if re.search(pattern, lowered_query):
                            canonical_ids.add(canonical_id)
                            break

                    if matched:
                        canonical_ids.add(canonical_id)

            except Exception as e:
                logger.warning("Error matching candidate entities in Neo4j: %s", e)

        return list(canonical_ids)

    def get_neighborhood(self, entity_ids: List[str]) -> Dict[str, Any]:
        """Query Neo4j for the 1-hop and 2-hop neighborhood of given entity IDs.

        Args:
            entity_ids: List of canonical entity IDs.

        Returns:
            Dict containing unique entities, relationships, papers, and mentions.
        """
        if not entity_ids:
            return {
                "entities": [],
                "relationships": [],
                "papers": [],
                "mentions": [],
            }

        # Step 1: Query 1-hop connections from start entity_ids
        cypher_hop = """
        MATCH (s:Entity)-[r]-(o:Entity)
        WHERE s.canonical_id IN $entity_ids
          AND type(r) <> "MENTIONS"
          AND r.confidence_score >= $min_confidence
        RETURN startNode(r).canonical_id AS source_id,
               startNode(r).name AS source_name,
               startNode(r).entity_type AS source_type,
               type(r) AS rel_type,
               r.confidence_score AS confidence_score,
               r.curation_status AS curation_status,
               r.id AS rel_id,
               endNode(r).canonical_id AS target_id,
               endNode(r).name AS target_name,
               endNode(r).entity_type AS target_type
        """

        unique_rels: Dict[tuple, Dict[str, Any]] = {}
        unique_entities: Dict[str, Dict[str, Any]] = {}

        def process_rel_records(records: List[Any]) -> Set[str]:
            nodes_involved = set()
            for r in records:
                try:
                    rec = dict(r)
                except (ValueError, TypeError):
                    rec = r

                src_id = rec.get("source_id")
                tgt_id = rec.get("target_id")
                rel_type = rec.get("rel_type")
                conf = rec.get("confidence_score")
                curation = rec.get("curation_status")
                rel_id = rec.get("rel_id")

                if not src_id or not tgt_id or not rel_type:
                    continue

                # Deduplicate relationships by key (rel_id if exists, otherwise source, target, type)
                rel_key = (src_id, tgt_id, rel_type)
                if rel_key not in unique_rels:
                    unique_rels[rel_key] = {
                        "id": rel_id,
                        "source_id": src_id,
                        "source_name": rec.get("source_name"),
                        "source_type": rec.get("source_type"),
                        "target_id": tgt_id,
                        "target_name": rec.get("target_name"),
                        "target_type": rec.get("target_type"),
                        "rel_type": rel_type,
                        "confidence_score": conf,
                        "curation_status": curation,
                    }

                # Collect entities
                for prefix in ("source", "target"):
                    ent_id = rec.get(f"{prefix}_id")
                    if ent_id and ent_id not in unique_entities:
                        unique_entities[ent_id] = {
                            "canonical_id": ent_id,
                            "name": rec.get(f"{prefix}_name"),
                            "entity_type": rec.get(f"{prefix}_type"),
                        }
                    if ent_id:
                        nodes_involved.add(ent_id)
            return nodes_involved

        # First hop
        try:
            records_hop1 = self.neo4j_client.execute_query(
                cypher_hop,
                {"entity_ids": entity_ids, "min_confidence": self.min_confidence},
            )
            hop1_nodes = process_rel_records(records_hop1)
        except Exception as e:
            logger.warning("Error fetching hop 1 relationships: %s", e)
            hop1_nodes = set()

        # Second hop (using nodes discovered in hop 1 + original start nodes)
        all_hop1_ids = list(hop1_nodes.union(entity_ids))
        if all_hop1_ids:
            try:
                records_hop2 = self.neo4j_client.execute_query(
                    cypher_hop,
                    {"entity_ids": all_hop1_ids, "min_confidence": self.min_confidence},
                )
                process_rel_records(records_hop2)
            except Exception as e:
                logger.warning("Error fetching hop 2 relationships: %s", e)

        # Include start entity nodes in unique_entities even if they have no relationships
        try:
            entity_details = self.neo4j_client.execute_query(
                "MATCH (e:Entity) WHERE e.canonical_id IN $entity_ids RETURN e.canonical_id AS canonical_id, e.name AS name, e.entity_type AS entity_type",
                {"entity_ids": entity_ids},
            )
            for r in entity_details:
                try:
                    rec = dict(r)
                except (ValueError, TypeError):
                    rec = r
                cid = rec.get("canonical_id")
                if cid and cid not in unique_entities:
                    unique_entities[cid] = {
                        "canonical_id": cid,
                        "name": rec.get("name"),
                        "entity_type": rec.get("entity_type"),
                    }
        except Exception as e:
            logger.warning("Error fetching start entity details: %s", e)

        # Collect papers and mentions for all entities in the neighborhood
        all_neighborhood_ids = list(unique_entities.keys())
        unique_papers: Dict[str, Dict[str, Any]] = {}
        mentions: List[Dict[str, Any]] = []

        if all_neighborhood_ids:
            cypher_papers = """
            MATCH (p:Paper)-[m:MENTIONS]->(e:Entity)
            WHERE e.canonical_id IN $entity_ids
            RETURN p.doi AS doi,
                   p.pmid AS pmid,
                   p.title AS title,
                   p.journal AS journal,
                   p.published_date AS published_date,
                   e.canonical_id AS entity_id,
                   m.confidence_score AS confidence_score
            """
            try:
                paper_records = self.neo4j_client.execute_query(
                    cypher_papers, {"entity_ids": all_neighborhood_ids}
                )
                for r in paper_records:
                    try:
                        rec = dict(r)
                    except (ValueError, TypeError):
                        rec = r

                    doi = rec.get("doi")
                    ent_id = rec.get("entity_id")
                    if not doi or not ent_id:
                        continue

                    # Deduplicate papers
                    if doi not in unique_papers:
                        unique_papers[doi] = {
                            "doi": doi,
                            "pmid": rec.get("pmid"),
                            "title": rec.get("title"),
                            "journal": rec.get("journal"),
                            "published_date": rec.get("published_date"),
                        }

                    # Add mention record
                    mentions.append(
                        {
                            "paper_doi": doi,
                            "entity_id": ent_id,
                            "confidence_score": rec.get("confidence_score"),
                        }
                    )
            except Exception as e:
                logger.warning("Error fetching papers/mentions: %s", e)

        return {
            "entities": list(unique_entities.values()),
            "relationships": list(unique_rels.values()),
            "papers": list(unique_papers.values()),
            "mentions": mentions,
        }

    def format_neighborhood_context(self, neighborhood: Dict[str, Any]) -> str:
        """Format the retrieved neighborhood subgraph into a clean textual summary.

        Args:
            neighborhood: Dict containing entities, relationships, papers, and mentions.

        Returns:
            Structured text block summarizing the graph context.
        """
        entities = neighborhood.get("entities") or []
        relationships = neighborhood.get("relationships") or []
        papers = neighborhood.get("papers") or []
        mentions = neighborhood.get("mentions") or []

        if not entities and not relationships:
            return ""

        # Map DOI to paper metadata dict
        doi_to_paper = {p["doi"]: p for p in papers}

        # Map entity_id to dict of paper_doi -> paper
        entity_to_papers: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for m in mentions:
            ent_id = m["entity_id"]
            paper_doi = m["paper_doi"]
            if paper_doi in doi_to_paper:
                if ent_id not in entity_to_papers:
                    entity_to_papers[ent_id] = {}
                entity_to_papers[ent_id][paper_doi] = doi_to_paper[paper_doi]

        lines = []

        # 1. Format identified entities
        if entities:
            lines.append("Identified Entities:")
            for ent in entities[:30]:
                lines.append(f"- {ent['name']} ({ent['entity_type']})")
            if len(entities) > 30:
                lines.append(f"... and {len(entities) - 30} more entities.")

        # 2. Format High-Confidence relationships
        if relationships:
            if lines and lines[-1] != "":
                lines.append("")
            lines.append("Knowledge Graph Relations:")
            # Sort by confidence descending
            sorted_rels = sorted(relationships, key=lambda x: x.get("confidence_score") or 0.0, reverse=True)
            for rel in sorted_rels[:50]:
                src_name = rel["source_name"]
                src_type = rel["source_type"]
                tgt_name = rel["target_name"]
                tgt_type = rel["target_type"]
                rel_type = rel["rel_type"]
                conf = rel["confidence_score"] or 0.0

                src_id = rel["source_id"]
                tgt_id = rel["target_id"]

                # Find co-mentioning papers
                src_papers = entity_to_papers.get(src_id, {})
                tgt_papers = entity_to_papers.get(tgt_id, {})
                co_dois = set(src_papers.keys()) & set(tgt_papers.keys())

                rel_label = rel_type.lower().replace("_", " ")

                if co_dois:
                    paper_strs = []
                    # Limit to top 3 co-mentioning papers
                    for doi in list(co_dois)[:3]:
                        paper = src_papers[doi]
                        pmid_part = f", PMID: {paper['pmid']}" if paper.get("pmid") else ""
                        paper_strs.append(f"DOI: {doi}{pmid_part}")
                    if len(co_dois) > 3:
                        paper_strs.append(f"and {len(co_dois) - 3} more")
                    papers_text = " and ".join(paper_strs)
                    lines.append(
                        f"- {src_name} ({src_type}) {rel_label} {tgt_name} ({tgt_type}) "
                        f"[confidence: {conf:.2f}] in Paper ({papers_text})."
                    )
                else:
                    lines.append(
                        f"- {src_name} ({src_type}) {rel_label} {tgt_name} ({tgt_type}) "
                        f"[confidence: {conf:.2f}]."
                    )
            if len(relationships) > 50:
                lines.append(f"... and {len(relationships) - 50} more relations.")

        # 3. Format overall entity annotations & references
        entity_mentions = []
        mention_count = 0
        for ent in entities:
            ent_id = ent["canonical_id"]
            ent_name = ent["name"]
            ent_type = ent["entity_type"]

            papers_for_ent = entity_to_papers.get(ent_id, {})
            if papers_for_ent:
                if mention_count >= 10:
                    continue
                paper_strs = []
                # Limit to top 3 papers per entity
                for p in list(papers_for_ent.values())[:3]:
                    pmid_part = f", PMID: {p['pmid']}" if p.get("pmid") else ""
                    title_part = f"'{p['title']}' " if p.get("title") else ""
                    paper_strs.append(f"{title_part}(DOI: {p['doi']}{pmid_part})")
                if len(papers_for_ent) > 3:
                    paper_strs.append(f"and {len(papers_for_ent) - 3} more papers")
                entity_mentions.append(
                    f"- {ent_name} ({ent_type}) is mentioned in: {'; '.join(paper_strs)}"
                )
                mention_count += 1

        if entity_mentions:
            if lines:
                lines.append("")
            lines.append("Entity Literature Mentions:")
            lines.extend(entity_mentions)
            if len(entities) > mention_count:
                # Indicate some entities were omitted if we hit the limit
                has_omitted = False
                for ent in entities[mention_count:]:
                    if ent["canonical_id"] in entity_to_papers:
                        has_omitted = True
                        break
                if has_omitted:
                    lines.append("... and more literature mentions.")

        return "\n".join(lines)

    def retrieve_graph_context(
        self, query: Optional[str] = None, entity_ids: Optional[List[str]] = None
    ) -> str:
        """Retrieve and format local neighborhood graph context.

        Args:
            query: Search query string.
            entity_ids: Optional list of pre-resolved entity IDs.

        Returns:
            Text block of graph context, or empty string.
        """
        resolved_ids = []
        if entity_ids:
            resolved_ids.extend(entity_ids)

        if query:
            extracted = self.extract_candidates(query)
            for eid in extracted:
                if eid not in resolved_ids:
                    resolved_ids.append(eid)

        if not resolved_ids:
            return ""

        neighborhood = self.get_neighborhood(resolved_ids)
        return self.format_neighborhood_context(neighborhood)


def retrieve_graph_context(
    query: str,
    neo4j_client: Optional[Neo4jClient] = None,
    min_confidence: float = 0.70,
    synonym_resolver: Optional[Any] = None,
    ner_extractor: Optional[Any] = None,
) -> str:
    """Helper module-level function for retrieving graph context.

    Instantiates GraphRetriever and performs the lookup.

    Args:
        query: User search query.
        neo4j_client: Optional Neo4jClient wrapper.
        min_confidence: Minimum confidence score threshold.
        synonym_resolver: Optional synonym resolver.
        ner_extractor: Optional NER extractor.

    Returns:
        Formatted textual summary of the local neighborhood.
    """
    retriever = GraphRetriever(
        neo4j_client=neo4j_client,
        min_confidence=min_confidence,
        synonym_resolver=synonym_resolver,
        ner_extractor=ner_extractor,
    )
    return retriever.retrieve_graph_context(query=query)
