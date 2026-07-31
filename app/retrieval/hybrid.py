"""Hybrid retriever module combining dense vector search and sparse full-text search."""

import os
from typing import Any

from app.retrieval.embeddings import EmbeddingClient
from app.retrieval.vector import LanceDBClient


class HybridRetriever:
    """
    Hybrid retriever combining dense semantic search and sparse lexical search,
    followed by cross-encoder reranking.
    """

    def __init__(
        self,
        vector_client: LanceDBClient | None = None,
        embedding_client: EmbeddingClient | None = None,
        reranker_model_name: str | None = None,
    ) -> None:
        """Initialize HybridRetriever.

        Args:
            vector_client: Optional LanceDBClient instance.
            embedding_client: Optional EmbeddingClient instance.
            reranker_model_name: Optional cross-encoder model name.
        """
        self.vector_client = vector_client or LanceDBClient()
        self.embedding_client = embedding_client or EmbeddingClient()
        self.reranker_model_name = reranker_model_name or os.getenv(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self._reranker = None
        self._reranker_failed = False

        # Attempt to initialize FTS index on 'content'
        self._ensure_fts_index()

    def _ensure_fts_index(self) -> None:
        """Ensure FTS index is created on the 'content' column of the LanceDB table."""
        try:
            table = self.vector_client.init_table()
            try:
                table.create_fts_index("content", exist_ok=True)
            except TypeError:
                # Fallback for LanceDB versions that do not support exist_ok
                table.create_fts_index("content")
        except Exception:
            # Silent fallback if database is empty or another error occurs during init.
            # It will be retried during retrieval if needed.
            pass

    @property
    def reranker(self) -> Any:
        """Lazy load the CrossEncoder model."""
        if self._reranker is None and not self._reranker_failed:
            try:
                from sentence_transformers import CrossEncoder

                # Share device with embedding client if available
                device = getattr(self.embedding_client, "device", None)
                if not device:
                    import torch
                    if torch.cuda.is_available():
                        device = "cuda"
                    elif torch.backends.mps.is_available():
                        device = "mps"
                    else:
                        device = "cpu"

                self._reranker = CrossEncoder(self.reranker_model_name, device=device)
            except Exception as e:
                print(
                    f"Warning: Failed to load CrossEncoder model "
                    f"'{self.reranker_model_name}'. Reranking will fall "
                    f"back to hybrid retrieval scores. Error: {e}"
                )
                self._reranker_failed = True
                self._reranker = None
        return self._reranker

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        alpha: float = 0.5,
        dense_limit: int | None = None,
        sparse_limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve top_k combined candidates using hybrid dense and sparse search.

        Args:
            query: The search query text.
            top_k: Number of final reranked candidates to return.
            fusion_method: Method to merge results, 'rrf' or 'norm'.
            rrf_k: RRF rank constant. Defaults to 60.
            alpha: Weight for dense scores in normalized combination.
            dense_limit: Max dense candidates before fusion.
            sparse_limit: Max sparse candidates before fusion.

        Returns:
            List of top_k dictionaries with keys:
            ['id', 'document_id', 'chunk_index', 'content',
             'pmid', 'doi', 'title', 'score']
        """
        if not query:
            return []

        # Default limits to fetch a larger pool for reranking
        d_limit = dense_limit or (top_k * 4)
        s_limit = sparse_limit or (top_k * 4)

        # 1. Dense Semantic Retrieval
        try:
            query_vector = self.embedding_client.embed_query(query)
            dense_results = self.vector_client.search_vector(
                query_vector, limit=d_limit
            )
        except Exception as e:
            print(f"Warning: Dense retrieval failed: {e}")
            dense_results = []

        # 2. Sparse Lexical Retrieval (FTS)
        try:
            table = self.vector_client.init_table()
            try:
                fts_results = (
                    table.search(query, query_type="fts")
                    .limit(s_limit)
                    .to_list()
                )
            except Exception:
                # Retry after ensuring index exists
                self._ensure_fts_index()
                try:
                    fts_results = (
                        table.search(query, query_type="fts")
                        .limit(s_limit)
                        .to_list()
                    )
                except Exception as e2:
                    print(f"Warning: LanceDB FTS search failed: {e2}")
                    fts_results = []
        except Exception as e:
            print(f"Warning: FTS query initialization failed: {e}")
            fts_results = []

        # Format sparse results uniformly
        sparse_results = []
        for item in fts_results:
            sparse_results.append({
                "id": item.get("id"),
                "document_id": item.get("document_id"),
                "chunk_index": item.get("chunk_index"),
                "content": item.get("content"),
                "pmid": item.get("pmid"),
                "doi": item.get("doi"),
                "title": item.get("title"),
                "score": (
                    item.get("_score")
                    if "_score" in item
                    else item.get("score", 0.0)
                ),
            })

        # 3. Candidate Fusion
        if fusion_method.lower() == "rrf":
            combined_candidates = self._reciprocal_rank_fusion(
                dense_results, sparse_results, k=rrf_k
            )
        else:
            combined_candidates = self._normalized_score_combination(
                dense_results, sparse_results, alpha=alpha
            )

        if not combined_candidates:
            return []

        # 4. Reranking
        reranker = self.reranker
        if reranker is not None and len(combined_candidates) > 0:
            try:
                pairs = [(query, c["content"]) for c in combined_candidates]
                scores = reranker.predict(pairs)
                for c, score in zip(combined_candidates, scores, strict=False):
                    c["score"] = float(score)
                # Sort by reranked score descending
                combined_candidates.sort(key=lambda x: x["score"], reverse=True)
            except Exception as e:
                print(f"Warning: Reranking prediction failed: {e}")

        # Return top_k candidates with requested keys
        output_keys = [
            "id",
            "document_id",
            "chunk_index",
            "content",
            "pmid",
            "doi",
            "title",
            "score",
        ]
        final_candidates = []
        for c in combined_candidates[:top_k]:
            final_candidates.append({k: c.get(k) for k in output_keys})

        return final_candidates

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[dict[str, Any]],
        sparse_results: list[dict[str, Any]],
        k: int,
    ) -> list[dict[str, Any]]:
        """Perform Reciprocal Rank Fusion (RRF) on dense and sparse candidates."""
        rrf_scores = {}
        candidates_map = {}

        # Rank is 1-based index in the results
        for rank, item in enumerate(dense_results):
            item_id = item.get("id") or item.get("content")
            if not item_id:
                continue
            rrf_scores[item_id] = (
                rrf_scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
            )
            if item_id not in candidates_map:
                candidates_map[item_id] = item

        for rank, item in enumerate(sparse_results):
            item_id = item.get("id") or item.get("content")
            if not item_id:
                continue
            rrf_scores[item_id] = (
                rrf_scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
            )
            if item_id not in candidates_map:
                candidates_map[item_id] = item

        sorted_ids = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
        )

        combined = []
        for item_id in sorted_ids:
            candidate = dict(candidates_map[item_id])
            candidate["score"] = rrf_scores[item_id]
            combined.append(candidate)

        return combined

    def _normalized_score_combination(
        self,
        dense_results: list[dict[str, Any]],
        sparse_results: list[dict[str, Any]],
        alpha: float,
    ) -> list[dict[str, Any]]:
        """Combine dense and sparse scores by min-max normalization."""
        def normalize(results: list[dict[str, Any]]) -> dict[str, float]:
            if not results:
                return {}
            scores = [r.get("score", 0.0) for r in results]
            min_s, max_s = min(scores), max(scores)
            denom = max_s - min_s
            if denom == 0.0:
                return {r.get("id") or r.get("content"): 1.0 for r in results}
            return {
                r.get("id") or r.get("content"): (
                    (r.get("score", 0.0) - min_s) / denom
                )
                for r in results
            }

        candidates_map = {}
        for item in dense_results + sparse_results:
            item_id = item.get("id") or item.get("content")
            if item_id and item_id not in candidates_map:
                candidates_map[item_id] = item

        dense_norm = normalize(dense_results)
        sparse_norm = normalize(sparse_results)

        combined_scores = {}
        for item_id in candidates_map:
            d_score = dense_norm.get(item_id, 0.0)
            s_score = sparse_norm.get(item_id, 0.0)
            combined_scores[item_id] = (
                alpha * d_score + (1.0 - alpha) * s_score
            )

        sorted_ids = sorted(
            combined_scores.keys(),
            key=lambda x: combined_scores[x],
            reverse=True,
        )

        combined = []
        for item_id in sorted_ids:
            candidate = dict(candidates_map[item_id])
            candidate["score"] = combined_scores[item_id]
            combined.append(candidate)

        return combined
