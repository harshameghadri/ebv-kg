"""
2-Hop Subgraph Neighborhood Pruner (SPOKE-inspired)

Ranks retrieved candidate graph paths/relationships using dense text embeddings
and cosine similarity against the user prompt, pruning candidate paths to top-K
items to fit within the LLM prompt context window.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from app.retrieval.fact_serializer import FactSerializer

logger = logging.getLogger(__name__)


class SubgraphPruner:
    """
    SPOKE-inspired Subgraph Neighborhood Pruner.
    
    Embeds candidate 1-hop and 2-hop graph paths/relationships, ranks them by
    semantic similarity to the user prompt, and prunes them to top-K paths or
    a specified token budget.
    """

    def __init__(
        self,
        embedding_client: Optional[Any] = None,
        model_name: Optional[str] = None,
        default_top_k: int = 10,
        similarity_threshold: Optional[float] = None,
        confidence_weight: float = 0.5,
    ) -> None:
        """
        Initialize SubgraphPruner.

        Args:
            embedding_client: Optional instance of EmbeddingClient or SentenceTransformer.
            model_name: Optional model name if instantiating EmbeddingClient lazily.
            default_top_k: Default maximum number of paths/relationships to retain.
            similarity_threshold: Optional minimum cosine similarity threshold.
            confidence_weight: Weight (0.0 to 1.0) for incorporating graph confidence_score.
                               composite_score = (1 - weight) * sim + weight * confidence
        """
        self._embedding_client = embedding_client
        self.model_name = model_name
        self.default_top_k = default_top_k
        self.similarity_threshold = similarity_threshold
        self.confidence_weight = max(0.0, min(1.0, confidence_weight))


    @property
    def embedding_client(self) -> Any:
        """Lazy load EmbeddingClient if not provided."""
        if self._embedding_client is None:
            from app.retrieval.embeddings import EmbeddingClient
            self._embedding_client = EmbeddingClient(model_name=self.model_name)
        return self._embedding_client

    def serialize_item(self, item: Any) -> str:
        """
        Convert a candidate path, relationship dict, or string item into a clean text triple for embedding.

        Args:
            item: String, relationship dict, multi-hop path list/dict, or custom object.

        Returns:
            Serialized natural language string representing the graph path/fact.
        """
        if isinstance(item, str):
            return item.strip()

        if isinstance(item, dict):
            # Case 1: Multi-hop path dict containing "path" key
            if "path" in item and isinstance(item["path"], list):
                return FactSerializer.format_multihop_path(item["path"])

            # Case 2: Relationship dictionary with source, target, relationship_type
            source = item.get("source_name") or item.get("source_id") or item.get("source")
            target = item.get("target_name") or item.get("target_id") or item.get("target")
            rel_type = (
                item.get("rel_type")
                or item.get("relationship_type")
                or item.get("predicate")
                or "ASSOCIATED_WITH"
            )
            source_type = item.get("source_type", "ENTITY")
            target_type = item.get("target_type", "ENTITY")

            if source and target:
                rel_label = str(rel_type).upper().replace("_", " ")
                return f"{source} ({source_type}) {rel_label} {target} ({target_type})"

            # Case 3: Fallback dict str
            return str(item)

        if isinstance(item, (list, tuple)):
            # Sequential multi-hop step list
            return FactSerializer.format_multihop_path(list(item))

        return str(item)

    def _get_embeddings(self, prompt: str, texts: List[str]) -> Tuple[List[float], List[List[float]]]:
        """
        Get vector embeddings for the prompt and a list of serialized path texts.
        Supports EmbeddingClient, SentenceTransformer, and custom duck-typed embedders.
        """
        if not prompt or not texts:
            return [], []

        client = self.embedding_client

        # 1. EmbeddingClient interface (verify returning iterable numeric array/list)
        if hasattr(client, "embed_query") and hasattr(client, "embed_documents"):
            try:
                prompt_emb = client.embed_query(prompt)
                docs_emb = client.embed_documents(texts)
                if isinstance(prompt_emb, (list, np.ndarray, tuple)) and isinstance(docs_emb, (list, np.ndarray, tuple)):
                    return prompt_emb, docs_emb
            except Exception:
                pass

        # 2. SentenceTransformer / FlagEmbedding / HuggingFace encode interface
        if hasattr(client, "encode"):
            prompt_emb_raw = client.encode(prompt, convert_to_numpy=True)
            if hasattr(prompt_emb_raw, "tolist"):
                prompt_emb = prompt_emb_raw.tolist()
            else:
                prompt_emb = list(prompt_emb_raw)

            docs_emb_raw = client.encode(texts, convert_to_numpy=True)
            if hasattr(docs_emb_raw, "tolist"):
                docs_emb = docs_emb_raw.tolist()
            else:
                docs_emb = [e.tolist() if hasattr(e, "tolist") else list(e) for e in docs_emb_raw]
            return prompt_emb, docs_emb

        raise ValueError(
            "Embedding client must implement embed_query/embed_documents or encode method."
        )


    def _compute_cosine_similarities(
        self, query_vec: List[float], doc_vecs: List[List[float]]
    ) -> List[float]:
        """Compute cosine similarity scores between query vector and doc vectors using NumPy."""
        if not query_vec or not doc_vecs:
            return []

        q = np.array(query_vec, dtype=np.float32)
        docs = np.array(doc_vecs, dtype=np.float32)

        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return [0.0] * len(doc_vecs)

        doc_norms = np.linalg.norm(docs, axis=1)
        # Avoid division by zero
        doc_norms = np.where(doc_norms == 0, 1e-10, doc_norms)

        dot_products = np.dot(docs, q)
        sims = dot_products / (q_norm * doc_norms)

        # Clip values to [-1.0, 1.0] for safety
        sims = np.clip(sims, -1.0, 1.0)
        return [float(s) for s in sims]

    def _extract_candidates(self, graph_context: Any) -> Tuple[List[Any], Optional[Dict[str, Any]]]:
        """Extract candidate path/relationship items and optional root neighborhood dict."""
        if graph_context is None:
            return [], None

        if isinstance(graph_context, dict):
            # If neighborhood context dict
            if "relationships" in graph_context and isinstance(graph_context["relationships"], list):
                return list(graph_context["relationships"]), graph_context
            # If single relationship dict
            if "source_id" in graph_context or "source_name" in graph_context or "source" in graph_context:
                return [graph_context], None
            return [], graph_context

        if isinstance(graph_context, list):
            return list(graph_context), None

        if isinstance(graph_context, str):
            lines = [line.strip() for line in graph_context.strip().split("\n") if line.strip()]
            return lines, None

        return [graph_context], None

    def prune(
        self,
        graph_context: Any,
        prompt: str,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        confidence_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Embeds candidate graph paths and prompt, ranks them by similarity, and prunes to top-K paths.

        Args:
            graph_context: Candidate items (dict neighborhood, list of rels/paths, or list of strings).
            prompt: User search query or prompt.
            top_k: Max paths to return (defaults to self.default_top_k).
            max_tokens: Optional maximum token context window limit.
            similarity_threshold: Optional min cosine similarity threshold.
            confidence_weight: Optional weight for hybrid similarity + confidence score ranking.

        Returns:
            Dict containing:
                - "pruned_items": List of top pruned items (ranked)
                - "scores": List of floats (composite similarity scores)
                - "formatted_context": Standardized text block of pruned graph context
                - "original_count": int
                - "pruned_count": int
                - "pruned_neighborhood": Optional updated neighborhood dict if input was a neighborhood dict
        """
        k = top_k if top_k is not None else self.default_top_k
        sim_thresh = (
            similarity_threshold
            if similarity_threshold is not None
            else self.similarity_threshold
        )
        conf_w = (
            confidence_weight
            if confidence_weight is not None
            else self.confidence_weight
        )
        conf_w = max(0.0, min(1.0, conf_w))

        candidates, root_neighborhood = self._extract_candidates(graph_context)
        original_count = len(candidates)

        if original_count == 0 or not prompt or k <= 0:
            return {
                "pruned_items": [],
                "scores": [],
                "formatted_context": "No structured knowledge graph paths retrieved for this query.",
                "original_count": original_count,
                "pruned_count": 0,
                "pruned_neighborhood": {
                    "entities": [],
                    "relationships": [],
                    "papers": [],
                    "mentions": [],
                }
                if root_neighborhood
                else None,
            }

        # 1. Serialize items for vector embedding
        serialized_texts = [self.serialize_item(item) for item in candidates]

        # 2. Get embeddings for prompt and candidate texts
        prompt_emb, docs_emb = self._get_embeddings(prompt, serialized_texts)

        # 3. Compute cosine similarities
        cosine_sims = self._compute_cosine_similarities(prompt_emb, docs_emb)

        # 4. Calculate composite score (combining similarity and relationship confidence if available)
        scored_candidates = []
        for idx, (item, text, sim) in enumerate(zip(candidates, serialized_texts, cosine_sims)):
            # Filter by similarity threshold if set
            if sim_thresh is not None and sim < sim_thresh:
                continue

            conf_score = 1.0
            if isinstance(item, dict):
                conf_val = item.get("confidence_score", item.get("confidence"))
                if conf_val is not None:
                    try:
                        conf_score = float(conf_val)
                    except (ValueError, TypeError):
                        conf_score = 1.0

            composite_score = (1.0 - conf_w) * sim + conf_w * conf_score
            scored_candidates.append({
                "item": item,
                "text": text,
                "similarity": sim,
                "score": composite_score,
                "original_idx": idx,
            })

        # 5. Rank candidates in descending order of composite score
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)

        # 6. Apply top-K cutoff and optional max_tokens token budget limit
        pruned_items = []
        scores = []
        accumulated_tokens = 0

        for candidate in scored_candidates:
            if len(pruned_items) >= k:
                break

            text = candidate["text"]
            # Rough token estimate: ~1 token per 4 characters or ~1.3 per word
            estimated_tokens = max(1, len(text) // 4)

            if max_tokens is not None and (accumulated_tokens + estimated_tokens) > max_tokens:
                if len(pruned_items) > 0:
                    break

            pruned_items.append(candidate["item"])
            scores.append(candidate["score"])
            accumulated_tokens += estimated_tokens

        # 7. Format pruned output context
        formatted_context = self.format_pruned_context(pruned_items)

        # 8. If input was a neighborhood dict, create a pruned neighborhood dict
        pruned_neighborhood = None
        if root_neighborhood and isinstance(root_neighborhood, dict):
            pruned_rels = [item for item in pruned_items if isinstance(item, dict)]
            pruned_entity_ids = set()
            for rel in pruned_rels:
                for key in ("source_id", "target_id", "source_canonical_id", "target_canonical_id"):
                    val = rel.get(key)
                    if val:
                        pruned_entity_ids.add(val)

            entities = root_neighborhood.get("entities", [])
            pruned_entities = [
                e for e in entities if e.get("canonical_id") in pruned_entity_ids
            ] if pruned_entity_ids else entities[:k]

            papers = root_neighborhood.get("papers", [])
            mentions = root_neighborhood.get("mentions", [])
            pruned_mentions = [
                m for m in mentions if m.get("entity_id") in pruned_entity_ids
            ] if pruned_entity_ids else mentions

            pruned_neighborhood = {
                "entities": pruned_entities,
                "relationships": pruned_rels,
                "papers": papers,
                "mentions": pruned_mentions,
            }

        return {
            "pruned_items": pruned_items,
            "scores": scores,
            "formatted_context": formatted_context,
            "original_count": original_count,
            "pruned_count": len(pruned_items),
            "pruned_neighborhood": pruned_neighborhood,
        }

    def format_pruned_context(self, pruned_items: List[Any]) -> str:
        """Format pruned items into a standardized natural language context block."""
        if not pruned_items:
            return "No structured knowledge graph paths retrieved for this query."

        # Check if items are relationship dicts
        if all(isinstance(x, dict) and ("source_name" in x or "source_id" in x or "source" in x) for x in pruned_items):
            return FactSerializer.format_subgraph_facts({"relationships": pruned_items}, max_facts=len(pruned_items))

        # Check if items are multi-hop paths (lists of step dicts or path dicts)
        lines = []
        for idx, item in enumerate(pruned_items, start=1):
            if isinstance(item, list):
                chain_str = FactSerializer.format_multihop_path(item)
                lines.append(f"[GRAPH PATH {idx}] {chain_str}")
            elif isinstance(item, dict) and "path" in item:
                chain_str = FactSerializer.format_multihop_path(item["path"])
                lines.append(f"[GRAPH PATH {idx}] {chain_str}")
            else:
                lines.append(f"[GRAPH FACT {idx}] {self.serialize_item(item)}")

        header = f"=== RETRIEVED KNOWLEDGE GRAPH FACT TRIPLES ({len(lines)} Facts) ==="
        return header + "\n" + "\n".join(lines)
