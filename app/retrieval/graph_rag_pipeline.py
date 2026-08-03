"""
Unified Graph-RAG Retrieval and Synthesis Pipeline.

Orchestrates:
1. Hybrid semantic search via LanceDB (LanceDBClient / EmbeddingClient / HybridRetriever).
2. Multi-hop graph retrieval via Neo4jClient or KuzuEngine (or GraphRetriever).
3. 2-Hop vector similarity path pruning via SubgraphPruner.
4. Fact serialization via FactSerializer.
5. Factual LLM answer synthesis via LLMClient (ClaudeSynthesisClient / LLMClient).

Returns structured JSON response containing:
- synthesized_answer
- pruned_facts
- text_chunks
- confidence_score
- dual_citations
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

from app.materialization.kuzu_engine import KuzuEngine
from app.materialization.neo4j_client import Neo4jClient
from app.retrieval.embeddings import EmbeddingClient
from app.retrieval.fact_serializer import FactSerializer
from app.retrieval.graph import GraphRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.subgraph_pruner import SubgraphPruner
from app.retrieval.vector import LanceDBClient
from app.synthesis.llm import ClaudeSynthesisClient

logger = logging.getLogger(__name__)


class GraphRAGPipeline:
    """
    Unified Graph-RAG Pipeline orchestrating hybrid vector search, multi-hop graph retrieval,
    vector path pruning, fact serialization, and LLM answer synthesis.
    """

    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        vector_client: Optional[LanceDBClient] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        graph_client: Optional[Union[Neo4jClient, KuzuEngine, Any]] = None,
        graph_retriever: Optional[GraphRetriever] = None,
        subgraph_pruner: Optional[SubgraphPruner] = None,
        llm_client: Optional[Any] = None,
        min_graph_confidence: float = 0.70,
        default_top_k_chunks: int = 5,
        default_top_k_facts: int = 10,
    ) -> None:
        """
        Initialize GraphRAGPipeline with configurable or default sub-components.

        Args:
            hybrid_retriever: Optional pre-configured HybridRetriever instance.
            vector_client: Optional LanceDBClient instance (used if hybrid_retriever is None).
            embedding_client: Optional EmbeddingClient instance.
            graph_client: Optional graph client (Neo4jClient, KuzuEngine, or duck-typed client).
            graph_retriever: Optional GraphRetriever instance.
            subgraph_pruner: Optional SubgraphPruner instance.
            llm_client: Optional LLM synthesis client instance.
            min_graph_confidence: Minimum confidence threshold for graph relationship traversal.
            default_top_k_chunks: Default number of hybrid text chunks to retrieve.
            default_top_k_facts: Default number of graph facts to prune and retain.
        """
        self.embedding_client = embedding_client or EmbeddingClient()
        self.vector_client = vector_client or LanceDBClient()

        if hybrid_retriever is not None:
            self.hybrid_retriever = hybrid_retriever
        else:
            self.hybrid_retriever = HybridRetriever(
                vector_client=self.vector_client,
                embedding_client=self.embedding_client,
            )

        self.graph_client = graph_client
        self.graph_retriever = graph_retriever
        self.min_graph_confidence = min_graph_confidence

        # If graph_retriever is not passed, attempt to build one if graph_client is Neo4jClient or None
        if self.graph_retriever is None:
            if isinstance(graph_client, Neo4jClient) or graph_client is None:
                try:
                    neo_client = (
                        graph_client
                        if isinstance(graph_client, Neo4jClient)
                        else Neo4jClient()
                    )
                    self.graph_retriever = GraphRetriever(
                        neo4j_client=neo_client,
                        min_confidence=self.min_graph_confidence,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to initialize default GraphRetriever with Neo4jClient: %s",
                        e,
                    )

        if subgraph_pruner is not None:
            self.subgraph_pruner = subgraph_pruner
        else:
            self.subgraph_pruner = SubgraphPruner(
                embedding_client=self.embedding_client,
                default_top_k=default_top_k_facts,
            )

        if llm_client is not None:
            self.llm_client = llm_client
        else:
            self.llm_client = ClaudeSynthesisClient()

        self.default_top_k_chunks = default_top_k_chunks
        self.default_top_k_facts = default_top_k_facts

    def _retrieve_graph_neighborhood(
        self, query: str, entity_ids: Optional[List[str]] = None
    ) -> Any:
        """
        Retrieve raw graph neighborhood context (relationships / nodes / paths) from Neo4j or KuzuEngine.

        Args:
            query: Search query prompt.
            entity_ids: Optional explicit list of canonical entity IDs.

        Returns:
            Dict or list containing graph neighborhood data, relationships, or paths.
        """
        resolved_ids: List[str] = list(entity_ids) if entity_ids else []

        # 1. Use explicit graph_retriever if available
        if self.graph_retriever is not None:
            if not resolved_ids and hasattr(self.graph_retriever, "extract_candidates"):
                try:
                    resolved_ids = self.graph_retriever.extract_candidates(query)
                except Exception as e:
                    logger.warning("GraphRetriever candidate extraction failed: %s", e)

            if hasattr(self.graph_retriever, "get_neighborhood"):
                try:
                    nb = self.graph_retriever.get_neighborhood(resolved_ids)
                    if nb and isinstance(nb, dict) and any(nb.values()):
                        return nb
                except Exception as e:
                    logger.warning("GraphRetriever get_neighborhood failed: %s", e)

            if hasattr(self.graph_retriever, "retrieve_graph_context"):
                try:
                    ctx = self.graph_retriever.retrieve_graph_context(
                        query=query, entity_ids=resolved_ids
                    )
                    if ctx:
                        return ctx
                except Exception as e:
                    logger.warning("GraphRetriever retrieve_graph_context failed: %s", e)

        # 2. Use graph_client directly if KuzuEngine or duck-typed
        if self.graph_client is not None:
            g_client = self.graph_client

            # Duck-typed get_neighborhood method
            if hasattr(g_client, "get_neighborhood"):
                try:
                    nb = g_client.get_neighborhood(resolved_ids or [query])
                    if nb:
                        return nb
                except Exception as e:
                    logger.warning("graph_client get_neighborhood failed: %s", e)

            # Duck-typed retrieve_graph_context method
            if hasattr(g_client, "retrieve_graph_context"):
                try:
                    ctx = g_client.retrieve_graph_context(
                        query=query, entity_ids=resolved_ids
                    )
                    if ctx:
                        return ctx
                except Exception as e:
                    logger.warning("graph_client retrieve_graph_context failed: %s", e)

            # KuzuEngine specific path retrieval
            if isinstance(g_client, KuzuEngine) or hasattr(
                g_client, "get_2hop_neighborhood"
            ):
                relationships = []
                entities = []
                try:
                    # If no resolved_ids, search for candidate entity nodes matching query terms
                    if not resolved_ids and hasattr(g_client, "execute_query"):
                        cypher = "MATCH (e:Entity) RETURN e.canonical_id AS canonical_id, e.name AS name, e.synonyms AS synonyms"
                        nodes = g_client.execute_query(cypher)
                        lowered_query = query.lower()
                        for n in nodes:
                            cid = n.get("canonical_id")
                            name = n.get("name") or ""
                            syns = n.get("synonyms") or []
                            terms = [name] + list(syns)
                            for t in terms:
                                if t and re.search(
                                    rf"\b{re.escape(t.lower())}\b", lowered_query
                                ):
                                    if cid and cid not in resolved_ids:
                                        resolved_ids.append(cid)
                                    break

                    for eid in resolved_ids:
                        if hasattr(g_client, "get_2hop_neighborhood"):
                            nb = g_client.get_2hop_neighborhood(eid)
                            rels = nb.get("relationships", [])
                            relationships.extend(rels)
                            entities.extend(nb.get("hop1_nodes", []))
                            entities.extend(nb.get("hop2_nodes", []))
                except Exception as e:
                    logger.warning("KuzuEngine neighborhood retrieval failed: %s", e)

                if relationships or entities:
                    return {
                        "entities": entities,
                        "relationships": relationships,
                        "papers": [],
                        "mentions": [],
                    }

            # Generic execute_query fallback for graph_client
            if hasattr(g_client, "execute_query"):
                try:
                    cypher = """
                    MATCH (s:Entity)-[r]-(o:Entity)
                    WHERE s.canonical_id IN $entity_ids OR toLower(s.name) CONTAINS toLower($query)
                    RETURN s.name AS source_name, s.canonical_id AS source_id, s.entity_type AS source_type,
                           type(r) AS rel_type, r.confidence AS confidence_score, r.curation_status AS curation_status,
                           o.name AS target_name, o.canonical_id AS target_id, o.entity_type AS target_type
                    """
                    records = g_client.execute_query(
                        cypher, {"entity_ids": resolved_ids, "query": query}
                    )
                    return {
                        "relationships": records,
                        "entities": [],
                        "papers": [],
                        "mentions": [],
                    }
                except Exception as e:
                    logger.warning("graph_client execute_query fallback failed: %s", e)

        return {"entities": [], "relationships": [], "papers": [], "mentions": []}

    def _format_pruned_facts(self, pruned_items: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert pruned graph items (relationships / paths / strings) into structured fact dictionaries.

        Args:
            pruned_items: List of pruned graph relationship dicts, path chains, or strings.

        Returns:
            List of structured fact dictionaries.
        """
        pruned_facts = []
        for idx, item in enumerate(pruned_items, start=1):
            fact_id = f"fact-{idx}"
            if isinstance(item, dict):
                # Relationship dictionary
                if "source_name" in item or "source_id" in item or "source" in item:
                    source = (
                        item.get("source_name")
                        or item.get("source_id")
                        or item.get("source", "Unknown")
                    )
                    source_type = item.get("source_type", "ENTITY")
                    target = (
                        item.get("target_name")
                        or item.get("target_id")
                        or item.get("target", "Unknown")
                    )
                    target_type = item.get("target_type", "ENTITY")
                    predicate = (
                        item.get("rel_type")
                        or item.get("relationship_type")
                        or item.get("type")
                        or item.get("predicate")
                        or "ASSOCIATED_WITH"
                    )
                    confidence = float(
                        item.get("confidence_score", item.get("confidence", 1.0))
                    )
                    curation_status = item.get("curation_status", "APPROVED")

                    serialized = FactSerializer.format_relationship_fact(
                        {
                            "source_name": source,
                            "source_type": source_type,
                            "target_name": target,
                            "target_type": target_type,
                            "relationship_type": predicate,
                            "confidence": confidence,
                            "curation_status": curation_status,
                            "evidence": item.get(
                                "evidence", item.get("evidence_text", "")
                            ),
                        },
                        index=idx,
                    )

                    pruned_facts.append(
                        {
                            "fact_id": fact_id,
                            "serialized": serialized,
                            "source": source,
                            "source_type": source_type,
                            "relationship": predicate,
                            "target": target,
                            "target_type": target_type,
                            "confidence": confidence,
                            "curation_status": curation_status,
                        }
                    )
                elif "path" in item:
                    serialized = FactSerializer.format_multihop_path(item["path"])
                    pruned_facts.append(
                        {
                            "fact_id": fact_id,
                            "serialized": f"[GRAPH PATH {idx}] {serialized}",
                            "path": item["path"],
                        }
                    )
                else:
                    serialized = str(item)
                    pruned_facts.append(
                        {
                            "fact_id": fact_id,
                            "serialized": f"[GRAPH FACT {idx}] {serialized}",
                            "raw": item,
                        }
                    )
            elif isinstance(item, (list, tuple)):
                serialized = FactSerializer.format_multihop_path(list(item))
                pruned_facts.append(
                    {
                        "fact_id": fact_id,
                        "serialized": f"[GRAPH PATH {idx}] {serialized}",
                        "path": list(item),
                    }
                )
            else:
                serialized = str(item)
                pruned_facts.append(
                    {
                        "fact_id": fact_id,
                        "serialized": f"[GRAPH FACT {idx}] {serialized}",
                    }
                )
        return pruned_facts

    def _build_dual_citations(
        self,
        llm_citations: List[Dict[str, Any]],
        text_chunks: List[Dict[str, Any]],
        pruned_facts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Build dual citations structure linking synthesized response to text chunks and graph facts.

        Args:
            llm_citations: Citations returned by the LLM synthesis client.
            text_chunks: Full list of hybrid semantic search text chunks.
            pruned_facts: List of structured pruned graph facts.

        Returns:
            Dict containing 'text_chunks', 'graph_facts', and 'all' citations.
        """
        text_chunk_citations = []
        for cit in llm_citations:
            src_idx = cit.get("source_index")
            chunk_id = cit.get("chunk_id")
            pmid = cit.get("pmid", "N/A")
            doi = cit.get("doi", "N/A")

            title = "N/A"
            content_snippet = ""
            if src_idx and 1 <= src_idx <= len(text_chunks):
                chunk_data = text_chunks[src_idx - 1]
                title = chunk_data.get("title", "N/A")
                raw_content = chunk_data.get("content", "")
                content_snippet = (
                    raw_content[:100] + "..."
                    if len(raw_content) > 100
                    else raw_content
                )

            text_chunk_citations.append(
                {
                    "type": "text_chunk",
                    "citation_index": src_idx,
                    "chunk_id": chunk_id,
                    "pmid": pmid,
                    "doi": doi,
                    "title": title,
                    "snippet": content_snippet,
                }
            )

        graph_fact_citations = []
        for idx, fact in enumerate(pruned_facts, start=1):
            graph_fact_citations.append(
                {
                    "type": "graph_fact",
                    "fact_index": idx,
                    "fact_id": fact.get("fact_id", f"fact-{idx}"),
                    "source": fact.get("source", "Unknown"),
                    "relationship": fact.get("relationship", "ASSOCIATED_WITH"),
                    "target": fact.get("target", "Unknown"),
                    "confidence": fact.get("confidence", 1.0),
                    "serialized": fact.get("serialized", ""),
                }
            )

        return {
            "text_chunks": text_chunk_citations,
            "graph_facts": graph_fact_citations,
            "all": text_chunk_citations + graph_fact_citations,
        }

    def query(
        self,
        query: str,
        top_k_chunks: Optional[int] = None,
        top_k_facts: Optional[int] = None,
        entity_ids: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        confidence_weight: Optional[float] = None,
        fusion_method: str = "rrf",
    ) -> Dict[str, Any]:
        """
        Execute the unified Graph-RAG pipeline.

        1. Hybrid semantic search via LanceDB
        2. Multi-hop graph retrieval via Neo4jClient or KuzuEngine
        3. 2-hop vector similarity path pruning via SubgraphPruner
        4. Fact serialization via FactSerializer
        5. Factual LLM answer synthesis via LLMClient

        Args:
            query: The user prompt / search query string.
            top_k_chunks: Max hybrid text chunks to retrieve (defaults to default_top_k_chunks).
            top_k_facts: Max graph facts to prune and retain (defaults to default_top_k_facts).
            entity_ids: Optional explicit candidate entity IDs for graph lookup.
            max_tokens: Optional token context window limit for pruned graph context.
            similarity_threshold: Optional cosine similarity threshold for graph path pruning.
            confidence_weight: Optional weight (0.0 to 1.0) combining similarity and graph confidence.
            fusion_method: Hybrid search fusion method ('rrf' or 'norm').

        Returns:
            Structured dictionary containing:
            - "synthesized_answer": str
            - "pruned_facts": List[Dict[str, Any]]
            - "text_chunks": List[Dict[str, Any]]
            - "confidence_score": float
            - "dual_citations": Dict[str, Any]
        """
        k_chunks = (
            top_k_chunks if top_k_chunks is not None else self.default_top_k_chunks
        )
        k_facts = (
            top_k_facts if top_k_facts is not None else self.default_top_k_facts
        )

        if not query or not query.strip():
            empty_dual = {"text_chunks": [], "graph_facts": [], "all": []}
            return {
                "synthesized_answer": "I do not know",
                "pruned_facts": [],
                "text_chunks": [],
                "confidence_score": 0.0,
                "dual_citations": empty_dual,
            }

        # Step 1: Hybrid Semantic Search via LanceDB
        logger.info("Executing Step 1: Hybrid semantic search for query '%s'", query)
        try:
            text_chunks = self.hybrid_retriever.retrieve(
                query=query,
                top_k=k_chunks,
                fusion_method=fusion_method,
            )
        except Exception as e:
            logger.warning("Hybrid retrieval encountered error: %s", e)
            text_chunks = []

        # Step 2: Multi-Hop Graph Retrieval via Neo4jClient or KuzuEngine
        logger.info("Executing Step 2: Multi-hop graph retrieval")
        raw_graph_context = self._retrieve_graph_neighborhood(
            query=query, entity_ids=entity_ids
        )

        # Step 3: 2-Hop Vector Similarity Path Pruning via SubgraphPruner
        logger.info("Executing Step 3: 2-hop vector similarity path pruning")
        prune_result = self.subgraph_pruner.prune(
            graph_context=raw_graph_context,
            prompt=query,
            top_k=k_facts,
            max_tokens=max_tokens,
            similarity_threshold=similarity_threshold,
            confidence_weight=confidence_weight,
        )

        pruned_items = prune_result.get("pruned_items", [])
        formatted_graph_context = prune_result.get(
            "formatted_context",
            "No structured knowledge graph paths retrieved for this query.",
        )

        # Step 4: Fact Serialization via FactSerializer
        logger.info("Executing Step 4: Fact serialization")
        pruned_facts = self._format_pruned_facts(pruned_items)

        # Step 5: Factual LLM Answer Synthesis via LLMClient
        logger.info("Executing Step 5: LLM answer synthesis")
        try:
            synthesis_res = self.llm_client.synthesize(
                query=query,
                retrieved_chunks=text_chunks,
                graph_context=formatted_graph_context,
            )
        except Exception as e:
            logger.error("LLM synthesis failed: %s", e)
            synthesis_res = {
                "answer": f"Error during answer synthesis: {e}",
                "confidence": 0.0,
                "citations": [],
            }

        synthesized_answer = synthesis_res.get("answer", "I do not know")
        confidence_score = float(synthesis_res.get("confidence", 0.0))
        llm_citations = synthesis_res.get("citations", [])

        # Construct Dual Citations
        dual_citations = self._build_dual_citations(
            llm_citations=llm_citations,
            text_chunks=text_chunks,
            pruned_facts=pruned_facts,
        )

        return {
            "synthesized_answer": synthesized_answer,
            "pruned_facts": pruned_facts,
            "text_chunks": text_chunks,
            "confidence_score": confidence_score,
            "dual_citations": dual_citations,
        }

    def run(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """Alias for query method."""
        return self.query(query=query, **kwargs)

    def query_json(self, query: str, indent: int = 2, **kwargs: Any) -> str:
        """
        Execute pipeline query and return response as formatted JSON string.

        Args:
            query: User query prompt string.
            indent: Indentation spaces for JSON formatting.
            **kwargs: Additional parameters passed to query().

        Returns:
            JSON string representation of the pipeline output.
        """
        result = self.query(query=query, **kwargs)
        return json.dumps(result, indent=indent)
