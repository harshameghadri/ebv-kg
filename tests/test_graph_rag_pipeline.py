"""
Unit tests for GraphRAGPipeline in app/retrieval/graph_rag_pipeline.py.

Verifies:
1. Pipeline initialization with default and custom components.
2. Hybrid semantic retrieval integration.
3. Multi-hop graph retrieval via Neo4jClient and KuzuEngine.
4. 2-hop vector similarity path pruning via SubgraphPruner.
5. Fact serialization via FactSerializer.
6. Factual LLM answer synthesis via LLMClient.
7. Return of structured JSON answer with:
   - synthesized_answer
   - pruned_facts
   - text_chunks
   - confidence_score
   - dual_citations
"""

import json
import os
import shutil
from unittest.mock import MagicMock, patch

import pytest

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

from app.materialization.kuzu_engine import KuzuEngine
from app.materialization.neo4j_client import Neo4jClient
from app.retrieval import GraphRAGPipeline
from app.retrieval.fact_serializer import FactSerializer
from app.retrieval.graph import GraphRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.subgraph_pruner import SubgraphPruner



class MockEmbeddingClient:
    """Mock embedding client returning deterministic vector embeddings."""

    def embed_query(self, text: str):
        return [0.1, 0.2, 0.3, 0.4]

    def embed_documents(self, texts: list[str]):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class MockLLMClient:
    """Mock LLM synthesis client."""

    def __init__(self, answer: str = "EBV LMP1 activates NF-kB pathway.", confidence: float = 0.95):
        self.answer = answer
        self.confidence = confidence

    def synthesize(self, query: str, retrieved_chunks: list, graph_context: str = ""):
        return {
            "answer": f"{self.answer} [1]",
            "confidence": self.confidence,
            "citations": [
                {
                    "source_index": 1,
                    "chunk_id": retrieved_chunks[0].get("id", "chunk-1") if retrieved_chunks else "chunk-1",
                    "pmid": retrieved_chunks[0].get("pmid", "12345") if retrieved_chunks else "12345",
                    "doi": retrieved_chunks[0].get("doi", "10.1000/test") if retrieved_chunks else "10.1000/test",
                }
            ] if retrieved_chunks else [],
        }


def test_pipeline_empty_query():
    """Verify empty query returns early with zero confidence and empty structure."""
    pipeline = GraphRAGPipeline(
        embedding_client=MockEmbeddingClient(),
        llm_client=MockLLMClient(),
    )
    result = pipeline.query("")
    assert result["synthesized_answer"] == "I do not know"
    assert result["confidence_score"] == 0.0
    assert result["pruned_facts"] == []
    assert result["text_chunks"] == []
    assert result["dual_citations"]["text_chunks"] == []
    assert result["dual_citations"]["graph_facts"] == []


def test_pipeline_full_orchestration():
    """Verify full 5-step pipeline execution with mock sub-components."""
    # 1. Mock HybridRetriever
    mock_hybrid = MagicMock(spec=HybridRetriever)
    mock_hybrid.retrieve.return_value = [
        {
            "id": "chunk-lmp1",
            "document_id": "doc-1",
            "chunk_index": 0,
            "content": "EBV LMP1 protein recruits TRAF molecules to activate NF-kB signaling.",
            "pmid": "998877",
            "doi": "10.1016/j.cell.2024.01.001",
            "title": "EBV LMP1 Signaling Mechanisms",
            "score": 0.92,
        }
    ]

    # 2. Mock GraphRetriever / Graph Context
    mock_graph_retriever = MagicMock(spec=GraphRetriever)
    mock_graph_retriever.extract_candidates.return_value = ["HGNC:6677"]
    mock_graph_retriever.get_neighborhood.return_value = {
        "entities": [
            {"canonical_id": "HGNC:6677", "name": "LMP1", "entity_type": "GENE"},
            {"canonical_id": "HGNC:7788", "name": "NFKB1", "entity_type": "GENE"},
        ],
        "relationships": [
            {
                "source_name": "LMP1",
                "source_type": "GENE",
                "target_name": "NFKB1",
                "target_type": "GENE",
                "rel_type": "ACTIVATES",
                "confidence_score": 0.98,
                "curation_status": "APPROVED",
            }
        ],
        "papers": [],
        "mentions": [],
    }

    # 3. SubgraphPruner
    pruner = SubgraphPruner(embedding_client=MockEmbeddingClient(), default_top_k=5)

    # 4. Mock LLM Synthesis Client
    mock_llm = MockLLMClient()

    # Construct Pipeline
    pipeline = GraphRAGPipeline(
        hybrid_retriever=mock_hybrid,
        graph_retriever=mock_graph_retriever,
        subgraph_pruner=pruner,
        llm_client=mock_llm,
        embedding_client=MockEmbeddingClient(),
    )

    result = pipeline.query("How does LMP1 activate NF-kB?")

    # Assertions
    assert "EBV LMP1 activates NF-kB pathway." in result["synthesized_answer"]
    assert result["confidence_score"] == 0.95
    assert len(result["text_chunks"]) == 1
    assert result["text_chunks"][0]["id"] == "chunk-lmp1"

    # Verify pruned facts
    assert len(result["pruned_facts"]) == 1
    fact = result["pruned_facts"][0]
    assert fact["source"] == "LMP1"
    assert fact["relationship"] == "ACTIVATES"
    assert fact["target"] == "NFKB1"
    assert fact["confidence"] == 0.98

    # Verify dual citations
    dual = result["dual_citations"]
    assert "text_chunks" in dual
    assert "graph_facts" in dual
    assert "all" in dual

    assert len(dual["text_chunks"]) == 1
    assert dual["text_chunks"][0]["chunk_id"] == "chunk-lmp1"
    assert dual["text_chunks"][0]["pmid"] == "998877"

    assert len(dual["graph_facts"]) == 1
    assert dual["graph_facts"][0]["source"] == "LMP1"
    assert dual["graph_facts"][0]["relationship"] == "ACTIVATES"


def test_pipeline_with_kuzu_engine():
    """Verify pipeline orchestration using KuzuEngine as graph_client."""
    mock_hybrid = MagicMock()
    mock_hybrid.retrieve.return_value = [
        {"id": "c1", "content": "EBNA1 binds origin of plasmid replication.", "pmid": "111", "doi": "10.1000/1"}
    ]

    # Create mock KuzuEngine
    kuzu = KuzuEngine(force_mock=True)
    kuzu.init_schema()
    kuzu.bulk_upsert_nodes(
        "Entity",
        [
            {"canonical_id": "EBNA1", "name": "EBNA1", "entity_type": "GENE"},
            {"canonical_id": "OriP", "name": "OriP", "entity_type": "GENIC_REGION"},
        ],
    )
    kuzu.bulk_upsert_edges(
        "ASSOCIATED_WITH",
        [
            {
                "source_id": "EBNA1",
                "target_id": "OriP",
                "relationship_type": "BINDS_TO",
                "confidence": 0.96,
            }
        ],
    )

    pipeline = GraphRAGPipeline(
        hybrid_retriever=mock_hybrid,
        graph_client=kuzu,
        embedding_client=MockEmbeddingClient(),
        llm_client=MockLLMClient(answer="EBNA1 binds OriP to maintain viral episome."),
    )

    result = pipeline.query("Does EBNA1 bind OriP?", entity_ids=["EBNA1"])

    assert "EBNA1 binds OriP" in result["synthesized_answer"]
    assert len(result["text_chunks"]) == 1
    assert len(result["pruned_facts"]) >= 1
    assert result["pruned_facts"][0]["source"] == "EBNA1"
    assert result["pruned_facts"][0]["target"] == "OriP"


def test_pipeline_with_neo4j_client():
    """Verify pipeline graph context retrieval with Neo4jClient."""
    mock_neo4j = MagicMock(spec=Neo4jClient)
    mock_neo4j.execute_query.return_value = [
        {
            "source_id": "CD21",
            "source_name": "CD21",
            "source_type": "PROTEIN",
            "rel_type": "RECEPTOR_FOR",
            "confidence_score": 0.99,
            "curation_status": "APPROVED",
            "target_id": "gp350",
            "target_name": "gp350",
            "target_type": "PROTEIN",
            "rel_id": "rel-123",
        }
    ]

    mock_hybrid = MagicMock()
    mock_hybrid.retrieve.return_value = [
        {"id": "c2", "content": "gp350 binds CD21 for B cell infection", "pmid": "222", "doi": "10.1000/2"}
    ]

    pipeline = GraphRAGPipeline(
        hybrid_retriever=mock_hybrid,
        graph_client=mock_neo4j,
        embedding_client=MockEmbeddingClient(),
        llm_client=MockLLMClient(answer="gp350 binds CD21."),
    )

    result = pipeline.query("What receptor does gp350 use?", entity_ids=["CD21"])
    assert result["confidence_score"] == 0.95
    assert len(result["pruned_facts"]) == 1
    assert result["pruned_facts"][0]["source"] == "CD21"
    assert result["pruned_facts"][0]["target"] == "gp350"


def test_pipeline_query_json():
    """Verify query_json returns a valid, formatted JSON string with required keys."""
    mock_hybrid = MagicMock()
    mock_hybrid.retrieve.return_value = [
        {"id": "chunk-json", "content": "EBV latent infection", "pmid": "555", "doi": "10.1000/json"}
    ]

    pipeline = GraphRAGPipeline(
        hybrid_retriever=mock_hybrid,
        embedding_client=MockEmbeddingClient(),
        llm_client=MockLLMClient(answer="EBV maintains latent infection in memory B cells."),
    )

    json_str = pipeline.query_json("How does EBV maintain latency?")
    parsed = json.loads(json_str)

    assert "synthesized_answer" in parsed
    assert "pruned_facts" in parsed
    assert "text_chunks" in parsed
    assert "confidence_score" in parsed
    assert "dual_citations" in parsed
    assert parsed["synthesized_answer"].startswith("EBV maintains latent infection")
