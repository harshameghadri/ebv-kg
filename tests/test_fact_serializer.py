import pytest
from app.retrieval.fact_serializer import FactSerializer

def test_format_relationship_fact():
    rel = {
        "source_name": "LMP1",
        "source_type": "GENE",
        "target_name": "NFKB1",
        "target_type": "GENE",
        "relationship_type": "ACTIVATES",
        "confidence": 0.95,
        "curation_status": "APPROVED",
        "evidence": "LMP1 CTAR1 domain activates NF-kB pathway."
    }
    fact = FactSerializer.format_relationship_fact(rel, index=1)
    assert "[GRAPH FACT 1]" in fact
    assert "LMP1 (GENE) ACTIVATES NFKB1 (GENE)" in fact
    assert "Conf: 0.95" in fact

def test_format_subgraph_facts():
    graph_context = {
        "relationships": [
            {
                "source_name": "EBNA1",
                "source_type": "GENE",
                "target_name": "p53",
                "target_type": "GENE",
                "relationship_type": "INHIBITS",
                "confidence": 0.90
            }
        ]
    }
    formatted = FactSerializer.format_subgraph_facts(graph_context)
    assert "RETRIEVED KNOWLEDGE GRAPH FACT TRIPLES" in formatted
    assert "EBNA1 (GENE) INHIBITS p53 (GENE)" in formatted

def test_format_multihop_path():
    path = [
        {"source": "LMP1", "predicate": "ACTIVATES", "target": "TRAF2"},
        {"source": "TRAF2", "predicate": "UPREGULATES", "target": "NFKB1"}
    ]
    formatted = FactSerializer.format_multihop_path(path)
    assert "LMP1 ──[ACTIVATES]──> TRAF2" in formatted
    assert "TRAF2 ──[UPREGULATES]──> NFKB1" in formatted
