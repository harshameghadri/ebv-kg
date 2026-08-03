"""
Unit tests for SubgraphPruner (2-Hop Subgraph Neighborhood Pruner).
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from app.retrieval.subgraph_pruner import SubgraphPruner
from app.retrieval.embeddings import EmbeddingClient


class DummyEmbeddingClient:
    """Mock embedding client returning deterministic normalized vectors for testing."""

    def __init__(self, mapping: dict = None):
        self.mapping = mapping or {}

    def embed_query(self, text: str) -> list[float]:
        return self._get_vector(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._get_vector(t) for t in texts]

    def _get_vector(self, text: str) -> list[float]:
        text_lower = text.lower()
        if text_lower in self.mapping:
            return self.mapping[text_lower]

        # Default vector based on length/hash
        vec = np.zeros(8, dtype=np.float32)
        if "lmp1" in text_lower or "nfkb" in text_lower:
            vec[0] = 1.0
            vec[1] = 0.5
        elif "ebna1" in text_lower or "orip" in text_lower:
            vec[2] = 1.0
            vec[3] = 0.5
        elif "burkitt" in text_lower or "lymphoma" in text_lower:
            vec[0] = 0.8
            vec[4] = 0.6
        else:
            vec[5] = 1.0

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


def test_subgraph_pruner_initialization():
    pruner = SubgraphPruner(default_top_k=5, similarity_threshold=0.3, confidence_weight=0.2)
    assert pruner.default_top_k == 5
    assert pruner.similarity_threshold == 0.3
    assert pruner.confidence_weight == 0.2


def test_serialize_item():
    pruner = SubgraphPruner()

    # 1. String item
    assert pruner.serialize_item("LMP1 ACTIVATES NFKB1") == "LMP1 ACTIVATES NFKB1"

    # 2. Relationship dictionary
    rel = {
        "source_name": "LMP1",
        "source_type": "GENE",
        "rel_type": "ACTIVATES",
        "target_name": "NFKB1",
        "target_type": "GENE",
    }
    serialized_rel = pruner.serialize_item(rel)
    assert "LMP1 (GENE) ACTIVATES NFKB1 (GENE)" in serialized_rel

    # 3. Multi-hop path dictionary
    multihop_dict = {
        "path": [
            {"source": "LMP1", "predicate": "ACTIVATES", "target": "TRAF2"},
            {"source": "TRAF2", "predicate": "REGULATES", "target": "NFKB1"},
        ]
    }
    serialized_path = pruner.serialize_item(multihop_dict)
    assert "LMP1" in serialized_path and "TRAF2" in serialized_path and "NFKB1" in serialized_path

    # 4. Multi-hop path list
    path_list = [
        {"source": "EBNA1", "predicate": "BINDS", "target": "OriP"}
    ]
    serialized_list = pruner.serialize_item(path_list)
    assert "EBNA1" in serialized_list and "OriP" in serialized_list


def test_prune_basic_ranking_and_top_k():
    mock_client = DummyEmbeddingClient()
    pruner = SubgraphPruner(embedding_client=mock_client, default_top_k=2)

    prompt = "How does LMP1 activate NFKB1 in Burkitt Lymphoma?"

    candidates = [
        {
            "source_name": "EBNA1",
            "source_type": "GENE",
            "rel_type": "BINDS",
            "target_name": "OriP",
            "target_type": "DNA",
            "confidence_score": 0.90,
        },
        {
            "source_name": "LMP1",
            "source_type": "GENE",
            "rel_type": "ACTIVATES",
            "target_name": "NFKB1",
            "target_type": "GENE",
            "confidence_score": 0.95,
        },
        {
            "source_name": "NFKB1",
            "source_type": "GENE",
            "rel_type": "ASSOCIATED_WITH",
            "target_name": "Burkitt Lymphoma",
            "target_type": "DISEASE",
            "confidence_score": 0.85,
        },
    ]

    res = pruner.prune(graph_context=candidates, prompt=prompt, top_k=2)

    assert res["original_count"] == 3
    assert res["pruned_count"] == 2
    assert len(res["pruned_items"]) == 2
    assert len(res["scores"]) == 2

    # Top item should be LMP1 ACTIVATES NFKB1 or NFKB1 ASSOCIATED_WITH Burkitt Lymphoma
    top_rel = res["pruned_items"][0]
    assert top_rel["source_name"] in ("LMP1", "NFKB1")
    assert "=== RETRIEVED KNOWLEDGE GRAPH FACT TRIPLES" in res["formatted_context"]


def test_prune_neighborhood_dict():
    mock_client = DummyEmbeddingClient()
    pruner = SubgraphPruner(embedding_client=mock_client, default_top_k=2)

    neighborhood = {
        "entities": [
            {"canonical_id": "HGNC:6642", "name": "LMP1", "entity_type": "GENE"},
            {"canonical_id": "HGNC:7794", "name": "NFKB1", "entity_type": "GENE"},
            {"canonical_id": "HGNC:3123", "name": "EBNA1", "entity_type": "GENE"},
        ],
        "relationships": [
            {
                "source_id": "HGNC:6642",
                "source_name": "LMP1",
                "source_type": "GENE",
                "rel_type": "ACTIVATES",
                "target_id": "HGNC:7794",
                "target_name": "NFKB1",
                "target_type": "GENE",
                "confidence_score": 0.95,
            },
            {
                "source_id": "HGNC:3123",
                "source_name": "EBNA1",
                "source_type": "GENE",
                "rel_type": "BINDS",
                "target_id": "HGNC:9999",
                "target_name": "OriP",
                "target_type": "DNA",
                "confidence_score": 0.80,
            },
        ],
        "papers": [],
        "mentions": [],
    }

    prompt = "How does LMP1 activate NFKB1?"
    res = pruner.prune(graph_context=neighborhood, prompt=prompt, top_k=1)

    assert res["original_count"] == 2
    assert res["pruned_count"] == 1
    assert res["pruned_items"][0]["source_name"] == "LMP1"

    p_neigh = res["pruned_neighborhood"]
    assert p_neigh is not None
    assert len(p_neigh["relationships"]) == 1
    # Check that entities include LMP1 and NFKB1
    pruned_cids = [e["canonical_id"] for e in p_neigh["entities"]]
    assert "HGNC:6642" in pruned_cids or "HGNC:7794" in pruned_cids


def test_similarity_threshold_filtering():
    # Prompt vector = [1, 0, 0]
    # Doc 1 vector = [1, 0, 0] (sim = 1.0)
    # Doc 2 vector = [0, 1, 0] (sim = 0.0)
    mock_client = MagicMock()
    mock_client.embed_query.return_value = [1.0, 0.0, 0.0]
    mock_client.embed_documents.return_value = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    pruner = SubgraphPruner(embedding_client=mock_client)
    candidates = ["Fact 1: LMP1 activates NFKB1", "Fact 2: Unrelated information"]

    res = pruner.prune(
        graph_context=candidates,
        prompt="LMP1 NFKB1",
        similarity_threshold=0.5,
    )

    assert res["original_count"] == 2
    assert res["pruned_count"] == 1
    assert res["pruned_items"] == ["Fact 1: LMP1 activates NFKB1"]


def test_confidence_weighting():
    # Both docs have similarity 0.8
    # Doc 1 confidence 0.9, Doc 2 confidence 0.5
    mock_client = MagicMock()
    mock_client.embed_query.return_value = [1.0, 0.0]
    mock_client.embed_documents.return_value = [
        [0.8, 0.6],
        [0.8, 0.6],
    ]

    pruner = SubgraphPruner(embedding_client=mock_client, confidence_weight=0.5)

    rel1 = {
        "source_name": "A",
        "target_name": "B",
        "rel_type": "REL1",
        "confidence_score": 0.9,
    }
    rel2 = {
        "source_name": "C",
        "target_name": "D",
        "rel_type": "REL2",
        "confidence_score": 0.5,
    }

    res = pruner.prune(graph_context=[rel2, rel1], prompt="Test prompt", top_k=2)

    assert res["pruned_items"][0] == rel1
    assert res["scores"][0] > res["scores"][1]


def test_max_tokens_budget_truncation():
    mock_client = MagicMock()
    mock_client.embed_query.return_value = [1.0, 0.0]
    mock_client.embed_documents.return_value = [
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ]

    pruner = SubgraphPruner(embedding_client=mock_client)

    long_fact_1 = "A " * 100  # ~200 chars => ~50 tokens
    long_fact_2 = "B " * 100  # ~200 chars => ~50 tokens
    long_fact_3 = "C " * 100  # ~200 chars => ~50 tokens

    res = pruner.prune(
        graph_context=[long_fact_1, long_fact_2, long_fact_3],
        prompt="Test prompt",
        max_tokens=60,  # Should allow only 1 item
    )

    assert res["original_count"] == 3
    assert res["pruned_count"] == 1


def test_empty_inputs_and_edge_cases():
    pruner = SubgraphPruner()

    # 1. None graph_context
    res_none = pruner.prune(graph_context=None, prompt="Test prompt")
    assert res_none["pruned_count"] == 0

    # 2. Empty list context
    res_empty = pruner.prune(graph_context=[], prompt="Test prompt")
    assert res_empty["pruned_count"] == 0

    # 3. Empty prompt
    res_no_prompt = pruner.prune(graph_context=["Fact 1"], prompt="")
    assert res_no_prompt["pruned_count"] == 0

    # 4. top_k = 0
    res_zero_k = pruner.prune(graph_context=["Fact 1"], prompt="Test", top_k=0)
    assert res_zero_k["pruned_count"] == 0


def test_sentence_transformer_encode_interface():
    mock_model = MagicMock()
    mock_model.encode.side_effect = lambda texts, convert_to_numpy=True: (
        np.array([1.0, 0.0]) if isinstance(texts, str) else np.array([[1.0, 0.0] for _ in texts])
    )

    pruner = SubgraphPruner(embedding_client=mock_model)
    res = pruner.prune(graph_context=["Fact A", "Fact B"], prompt="Query", top_k=2)

    assert res["pruned_count"] == 2
    assert len(res["scores"]) == 2
