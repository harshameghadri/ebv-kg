"""
Path-to-Text Fact Serializer for Graph-RAG Prompts

Converts multi-hop Cypher graph paths and Neo4j subgraphs into standardized,
high-density Natural Language Fact Triples with provenance citations for LLM synthesis.
"""

import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class FactSerializer:
    """Serializes Graph Nodes, Relationships, and Paths into LLM-ready Natural Language Fact Triples."""

    @staticmethod
    def format_relationship_fact(rel: Dict[str, Any], index: int = 1) -> str:
        """
        Formats a single relationship dictionary into a structured Fact string.
        Example: [GRAPH FACT 1] LMP1 (GENE) ACTIVATES NFKB1 (GENE) [Confidence: 0.94, Status: APPROVED]
        """
        source = rel.get("source_name", rel.get("source_id", "Unknown"))
        source_type = rel.get("source_type", "ENTITY")
        target = rel.get("target_name", rel.get("target_id", "Unknown"))
        target_type = rel.get("target_type", "ENTITY")
        predicate = rel.get("relationship_type", "ASSOCIATED_WITH")
        confidence = rel.get("confidence", 1.0)
        status = rel.get("curation_status", "APPROVED")
        evidence = rel.get("evidence", "")

        fact_str = f"[GRAPH FACT {index}] {source} ({source_type}) {predicate} {target} ({target_type}) [Conf: {confidence}, Status: {status}]"
        if evidence:
            fact_str += f"\n   Evidence Citation: \"{evidence[:150]}...\""
        return fact_str

    @staticmethod
    def format_subgraph_facts(graph_context: Dict[str, Any], max_facts: int = 20) -> str:
        """
        Formats a complete graph context dictionary into a bulleted list of Fact Triples.
        """
        relationships = graph_context.get("relationships", [])
        if not relationships:
            return "No structured knowledge graph paths retrieved for this query."

        fact_lines = []
        for idx, rel in enumerate(relationships[:max_facts], start=1):
            fact_lines.append(FactSerializer.format_relationship_fact(rel, index=idx))

        header = f"=== RETRIEVED KNOWLEDGE GRAPH FACT TRIPLES ({len(fact_lines)} Facts) ==="
        return header + "\n" + "\n".join(fact_lines)

    @staticmethod
    def format_multihop_path(path: List[Dict[str, Any]]) -> str:
        """Formats a sequential 2-hop or 3-hop graph path into a readable chain."""
        if not path:
            return ""
        
        segments = []
        for step in path:
            src = step.get("source", "Node")
            rel = step.get("predicate", "--")
            tgt = step.get("target", "Node")
            segments.append(f"{src} ──[{rel}]──> {tgt}")
            
        return " ──> ".join(segments)
