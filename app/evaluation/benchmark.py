"""Embedding Evaluation Suite for EBV Knowledge System.

This module provides the RAGEvaluator class to evaluate retrieval performance of
embedding models using gold-standard biological queries and document chunks.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional
import numpy as np

# Set up logger
logger = logging.getLogger(__name__)

# Default list of 10+ gold-standard EBV biological queries with expected DOIs/PMIDs
# and text content containing relevant keywords for retrieval evaluation.
DEFAULT_GOLDEN_QUERIES = [
    {
        "query": "How does Epstein-Barr virus EBNA1 bind to the plasmid origin of replication (oriP)?",
        "pmids": ["12300001"],
        "dois": ["10.1128/jvi.1"],
        "content": "Epstein-Barr virus nuclear antigen 1 (EBNA1) binds specifically to the family of repeats and dyad symmetry elements within the plasmid origin of replication (oriP), facilitating viral DNA replication."
    },
    {
        "query": "Role of LMP1 in activating NF-kappaB pathway in nasopharyngeal carcinoma cells.",
        "pmids": ["12300002"],
        "dois": ["10.1128/jvi.2"],
        "content": "The latent membrane protein 1 (LMP1) of EBV acts as a constitutively active tumor necrosis factor receptor (TNFR) mimic, triggering the NF-kappaB signaling pathway in nasopharyngeal carcinoma."
    },
    {
        "query": "Association between EBV infection and Multiple Sclerosis pathogenesis.",
        "pmids": ["12300003"],
        "dois": ["10.1128/jvi.3"],
        "content": "Epidemiological studies indicate a strong association between Epstein-Barr virus (EBV) infection and Multiple Sclerosis (MS) pathogenesis, potentially driven by cross-reactive antibodies (molecular mimicry) against glial antigens."
    },
    {
        "query": "What is the function of BZLF1 (ZEBRA) in triggering EBV lytic cycle reactivation?",
        "pmids": ["12300004"],
        "dois": ["10.1128/jvi.4"],
        "content": "BZLF1, also known as ZEBRA or EB1, is an immediate-early gene product of EBV that acts as a transcription factor to initiate the switch from latent infection to the lytic cycle."
    },
    {
        "query": "How does EBV GP350 glycoprotein mediate entry into host B lymphocytes via CD21?",
        "pmids": ["12300005"],
        "dois": ["10.1128/jvi.5"],
        "content": "The major viral envelope glycoprotein gp350/220 binds to the cellular receptor CD21 (CR2) on B lymphocytes, mediating the initial attachment and entry of Epstein-Barr virus."
    },
    {
        "query": "Epstein-Barr virus EBNA2 transcription factor activation of cellular MYC oncogene.",
        "pmids": ["12300006"],
        "dois": ["10.1128/jvi.6"],
        "content": "EBNA2 functions as a transcriptional activator that targets both viral promoters and cellular genes, including the direct up-regulation of the cellular oncogene c-MYC."
    },
    {
        "query": "Role of EBV encoded microRNAs (BART and BHRF1) in preventing B cell apoptosis.",
        "pmids": ["12300007"],
        "dois": ["10.1128/jvi.7"],
        "content": "EBV expresses multiple microRNAs from the BART and BHRF1 regions that target pro-apoptotic cellular genes, thus promoting B-cell survival and facilitating persistent infection."
    },
    {
        "query": "How does LMP2A mimic BCR signaling to maintain B-cell latency?",
        "pmids": ["12300008"],
        "dois": ["10.1128/jvi.8"],
        "content": "Latent membrane protein 2A (LMP2A) contains an immunoreceptor tyrosine-based activation motif (ITAM) that mimics tonic B-cell receptor (BCR) signaling, promoting survival of latent B cells."
    },
    {
        "query": "EBV glycoprotein gH/gL complexes interaction with cellular integrins for epithelial cell entry.",
        "pmids": ["12300009"],
        "dois": ["10.1128/jvi.9"],
        "content": "While B cell entry requires gp350 and gp42, EBV entry into epithelial cells is mediated by the gH/gL glycoprotein complex interacting directly with cellular integrins (alphaVbeta6/8)."
    },
    {
        "query": "Mechanisms of immune evasion by EBV viral IL-10 homolog BCRF1.",
        "pmids": ["12300010"],
        "dois": ["10.1128/jvi.10"],
        "content": "The BCRF1 gene encodes a viral interleukin-10 (vIL-10) homolog that suppresses inflammatory cytokine production, MHC class II expression, and T-cell activation to evade host immunity."
    },
    {
        "query": "LMP1-mediated induction of matrix metalloproteinases (MMPs) in metastasis.",
        "pmids": ["12300011"],
        "dois": ["10.1128/jvi.11"],
        "content": "LMP1 induces the expression of matrix metalloproteinases like MMP-9, degrading extracellular matrix and promoting invasive and metastatic phenotypes in nasopharyngeal carcinoma."
    }
]


class RAGEvaluator:
    """Evaluates RAG retrieval using embedding models on golden query datasets."""

    def __init__(
        self,
        queries: Optional[List[Dict[str, Any]]] = None,
        query_file_path: Optional[str] = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            queries: Optional list of query dicts containing 'query', 'pmids', 'dois',
                and optional 'content'.
            query_file_path: Optional path to a JSON file containing the queries.
        """
        if queries is not None:
            self.queries = queries
        elif query_file_path and os.path.exists(query_file_path):
            with open(query_file_path, "r", encoding="utf-8") as f:
                self.queries = json.load(f)
        else:
            self.queries = DEFAULT_GOLDEN_QUERIES

    def evaluate_embeddings(
        self, vector_client: Any, embedding_client: Any, k: int = 5
    ) -> Dict[str, Any]:
        """Evaluate the retrieval performance of an embedding client on a vector store.

        Args:
            vector_client: An instance of LanceDBClient (or a mock/compatible client).
            embedding_client: An instance of EmbeddingClient (or a mock/compatible client).
            k: The number of retrieved chunks to consider (default: 5).

        Returns:
            A dictionary containing:
              - 'mean_precision': Mean Precision@k
              - 'mean_recall': Mean Recall@k
              - 'mrr': Mean Reciprocal Rank@k
              - 'query_results': Detailed metrics for each query
        """
        precisions = []
        recalls = []
        reciprocal_ranks = []
        detailed_results = []

        for q_item in self.queries:
            query_text = q_item["query"]
            expected_pmids = set(q_item.get("pmids", []))
            expected_dois = set(q_item.get("dois", []))

            # Generate query embedding
            embedding = embedding_client.embed_query(query_text)

            # Search vector database
            retrieved = vector_client.search_vector(embedding, limit=k)

            # Calculate metrics
            relevant_retrieved_count = 0
            first_relevant_rank = None

            for idx, chunk in enumerate(retrieved):
                metadata = chunk.get("metadata") or {}
                pmid = chunk.get("pmid") or metadata.get("pmid")
                doi = chunk.get("doi") or metadata.get("doi")

                is_relevant = (pmid in expected_pmids) or (doi in expected_dois)
                if is_relevant:
                    relevant_retrieved_count += 1
                    if first_relevant_rank is None:
                        first_relevant_rank = idx + 1  # 1-indexed rank

            precision = relevant_retrieved_count / k

            total_relevant_expected = max(len(expected_pmids), len(expected_dois))
            if total_relevant_expected > 0:
                recall = relevant_retrieved_count / total_relevant_expected
            else:
                recall = 0.0

            reciprocal_rank = (
                1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
            )

            precisions.append(precision)
            recalls.append(recall)
            reciprocal_ranks.append(reciprocal_rank)

            detailed_results.append(
                {
                    "query": query_text,
                    "precision": precision,
                    "recall": recall,
                    "mrr": reciprocal_rank,
                    "retrieved": [
                        {
                            "id": c.get("id"),
                            "pmid": c.get("pmid")
                            or (c.get("metadata") or {}).get("pmid"),
                            "doi": c.get("doi")
                            or (c.get("metadata") or {}).get("doi"),
                            "score": c.get("score"),
                        }
                        for c in retrieved
                    ],
                }
            )

        mean_precision = float(np.mean(precisions)) if precisions else 0.0
        mean_recall = float(np.mean(recalls)) if recalls else 0.0
        mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

        return {
            "mean_precision": mean_precision,
            "mean_recall": mean_recall,
            "mrr": mrr,
            "query_results": detailed_results,
        }

    def populate_benchmark_data(
        self, vector_client: Any, embedding_client: Any
    ) -> None:
        """Helper method to populate the vector store with chunks derived from benchmark queries.

        Args:
            vector_client: An instance of LanceDBClient.
            embedding_client: An instance of EmbeddingClient.
        """
        vector_client.clear_table()
        vector_client.init_table()

        chunks = []
        for idx, q_item in enumerate(self.queries):
            content = q_item.get(
                "content", f"This is mock content for query: {q_item['query']}"
            )
            pmids = q_item.get("pmids", [])
            dois = q_item.get("dois", [])

            pmid = pmids[0] if pmids else f"pmid-{idx}"
            doi = dois[0] if dois else f"doi-{idx}"

            chunks.append(
                {
                    "id": f"bench-chunk-{idx}",
                    "document_id": f"bench-doc-{idx}",
                    "chunk_index": 0,
                    "content": content,
                    "pmid": pmid,
                    "doi": doi,
                    "title": f"EBV Study {idx}",
                    "metadata": {
                        "pmid": pmid,
                        "doi": doi,
                        "title": f"EBV Study {idx}",
                    },
                }
            )

        # General biomedical distractors to make search realistic
        distractors = [
            "Epstein-Barr virus is a member of the herpesvirus family and one of the most common human viruses.",
            "Infectious mononucleosis is most commonly caused by Epstein-Barr virus infection.",
            "The genome of EBV is a double-stranded, linear DNA molecule of approximately 172 kilobase pairs.",
            "EBV was discovered in 1964 by Anthony Epstein, Yvonne Barr, and Bert Achong in Burkitt lymphoma cells.",
            "B lymphocytes are the primary target cells for EBV latent infection in vivo.",
            "T cells and NK cells can also be infected by EBV, leading to lymphoproliferative disorders.",
            "GP350 is a target for neutralizing antibody responses against EBV.",
            "LMP1 expression is regulated by EBNA2 and cellular transcription factors.",
            "The latent replication origin oriP consists of the family of repeats and the dyad symmetry element.",
            "Standard cell lines used to study EBV include lymphoblastoid cell lines established by in vitro transformation.",
        ]

        for d_idx, dist_content in enumerate(distractors):
            chunks.append(
                {
                    "id": f"distractor-chunk-{d_idx}",
                    "document_id": f"distractor-doc-{d_idx}",
                    "chunk_index": 0,
                    "content": dist_content,
                    "pmid": f"distractor-pmid-{d_idx}",
                    "doi": f"distractor-doi-{d_idx}",
                    "title": f"Distractor Study {d_idx}",
                    "metadata": {
                        "pmid": f"distractor-pmid-{d_idx}",
                        "doi": f"distractor-doi-{d_idx}",
                        "title": f"Distractor Study {d_idx}",
                    },
                }
            )

        # Generate dense embeddings
        texts = [c["content"] for c in chunks]
        embeddings = embedding_client.embed_documents(texts)

        # Match embeddings to chunks
        for chunk, emb in zip(chunks, embeddings):
            chunk["vector"] = emb

        vector_client.add_chunks(chunks)

    def run_comparison(
        self,
        model_names: List[str],
        db_uri: str = "data/lancedb_eval/",
        k: int = 5,
    ) -> Dict[str, Dict[str, Any]]:
        """Run a comparison between multiple embedding models.

        Args:
            model_names: List of model names to compare.
            db_uri: Directory path for temporary evaluation LanceDB database.
            k: The number of retrieved chunks to consider.

        Returns:
            A dictionary mapping each model name to its evaluation results.
        """
        from app.retrieval.embeddings import EmbeddingClient
        from app.retrieval.vector import LanceDBClient

        results = {}
        for model in model_names:
            logger.info(f"Running evaluation for model: {model}")

            # Initialize embedding client
            embedding_client = EmbeddingClient(model_name=model)

            # Determine vector dimension by generating a dummy embedding
            vector_dim = len(embedding_client.embed_query("dummy"))

            # Create a model-specific table name
            safe_model_name = (
                model.replace("/", "_").replace("-", "_").replace(".", "_").lower()
            )
            table_name = f"eval_{safe_model_name}"

            # Initialize vector client
            vector_client = LanceDBClient(
                uri=db_uri, table_name=table_name, vector_dim=vector_dim
            )

            # Populate data and run evaluation
            self.populate_benchmark_data(vector_client, embedding_client)
            eval_metrics = self.evaluate_embeddings(
                vector_client, embedding_client, k=k
            )

            results[model] = eval_metrics

            # Clean up the table after run
            vector_client.clear_table()

        return results


def main() -> None:
    """CLI entry point for running the embedding model comparison."""
    import argparse

    parser = argparse.ArgumentParser(
        description="EBV Knowledge System - Embedding Evaluation Suite"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all-MiniLM-L6-v2", "BAAI/bge-m3"],
        help="List of embedding models to compare.",
    )
    parser.add_argument(
        "--db-uri",
        default="data/lancedb_eval/",
        help="URI/path for the evaluation LanceDB.",
    )
    parser.add_argument(
        "--query-file", help="Path to a JSON file containing custom evaluation queries."
    )
    parser.add_argument(
        "--k", type=int, default=5, help="Recall/precision limit K."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output logging."
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    evaluator = RAGEvaluator(query_file_path=args.query_file)

    print("\n" + "=" * 71)
    print("EBV Knowledge System - Embedding Model Comparison")
    print("=" * 71)
    print(f"Comparing models: {args.models}")
    print(f"Number of queries: {len(evaluator.queries)}")
    print(f"Retrieval limit K: {args.k}")
    print(f"Evaluation database: {args.db_uri}")
    print("=" * 71 + "\n")

    try:
        comparison_results = evaluator.run_comparison(
            model_names=args.models, db_uri=args.db_uri, k=args.k
        )

        # Display results in a formatted table
        print(
            f"{'Model Name':<25} | {'Precision@' + str(args.k):<12} | {'Recall@' + str(args.k):<12} | {'MRR@' + str(args.k):<12}"
        )
        print("-" * 71)
        for model, metrics in comparison_results.items():
            print(
                f"{model:<25} | {metrics['mean_precision']:<12.4f} | {metrics['mean_recall']:<12.4f} | {metrics['mrr']:<12.4f}"
            )
        print("-" * 71 + "\n")

    except Exception as e:
        logger.error(f"Error during comparison: {e}")
        import sys

        sys.exit(1)


if __name__ == "__main__":
    main()
