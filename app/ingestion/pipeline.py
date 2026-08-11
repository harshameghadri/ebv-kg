"""Unified ETL Pipeline Orchestrator for PubMed scraping, parsing, mapping, indexing, and materializing."""

import logging
import os
import psycopg
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.ingestion.pubmed_scraper import PubMedScraper
from app.ingestion.pmc_parser import PMCXMLParser
from app.ingestion.pdf_extractor import PDFExtractor
from app.processing.ner_extractor import NERExtractor
from app.processing.entity_mapper import EntityMapper
from app.ingestion.embeddings_pipeline import EmbeddingsPipeline
from app.materialization.materializer import Materializer
from app.materialization.neo4j_client import Neo4jClient
from app.retrieval.vector import LanceDBClient
from app.retrieval.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)


class ETLPipeline:
    """Orchestrates the end-to-end ingestion and knowledge graph build pipeline."""

    def __init__(
        self,
        pg_dsn: Optional[str] = None,
        lancedb_uri: Optional[str] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        neo4j_database: Optional[str] = None,
        staging_dir: str = "data/staging",
    ) -> None:
        """Initialize the ETLPipeline with all database configuration parameters."""
        self.pg_dsn = pg_dsn or os.getenv("DATABASE_URL") or os.getenv("POSTGRES_DSN")
        self.lancedb_uri = lancedb_uri or os.getenv("LANCEDB_URI", "data/lancedb/")
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")
        self.neo4j_database = neo4j_database or os.getenv("NEO4J_DATABASE")
        staging_env = os.getenv("FAST_STAGING_DIR") or os.getenv("STAGING_DIR")

        if staging_env:
            self.staging_dir = Path(staging_env)
        else:
            self.staging_dir = Path(staging_dir)



    def run_etl_pipeline(self, query: str, max_articles: int = 5) -> Dict[str, Any]:
        """Execute the complete end-to-end ETL pipeline.

        Args:
            query: PubMed search term.
            max_articles: Max articles to query and fetch.

        Returns:
            Dict containing stats of the run.
        """
        logger.info("Starting run_etl_pipeline with query='%s', max_articles=%d", query, max_articles)

        # Ensure we have a valid PostgreSQL DSN
        if not self.pg_dsn:
            raise ValueError("PostgreSQL DSN must be provided or configured via environment variables.")

        # Step 1: Scrape PubMed
        logger.info("Step 1: Scraper initialization")
        scraper = PubMedScraper(staging_dir=self.staging_dir)
        scraper_res = scraper.scrape(query, max_results=max_articles)
        logger.info("Scraper found %d articles.", scraper_res.get("total_found", 0))

        # Step 2: Parse and Map Documents
        logger.info("Step 2: Parse and Map Documents")
        pmc_parser = PMCXMLParser()
        pdf_extractor = PDFExtractor()
        ner_extractor = NERExtractor()
        entity_mapper = EntityMapper()

        # Connect to PostgreSQL for writes
        pg_conn = psycopg.connect(self.pg_dsn)
        processed_docs_count = 0

        try:
            # Collect unique absolute file paths to process
            xml_paths = set(Path(p).resolve() for p in scraper_res.get("xml_saved", []))
            xml_dir = self.staging_dir / "xml"
            if xml_dir.exists():
                for p in xml_dir.glob("*.xml"):
                    xml_paths.add(p.resolve())

            pdf_paths = set()
            pdf_dir = self.staging_dir / "pdf"
            if pdf_dir.exists():
                for p in pdf_dir.glob("*.pdf"):
                    pdf_paths.add(p.resolve())
            for p in self.staging_dir.glob("*.pdf"):
                pdf_paths.add(p.resolve())

            metadata_paths = set(Path(p).resolve() for p in scraper_res.get("metadata_saved", []))
            metadata_dir = self.staging_dir / "metadata"
            if metadata_dir.exists():
                for p in metadata_dir.glob("*.json"):
                    metadata_paths.add(p.resolve())

            # Parse XML files
            for xml_path in xml_paths:
                logger.info("Parsing JATS XML file: %s", xml_path)
                try:
                    parsed = pmc_parser.parse(xml_path)
                    processed_docs_count += self._process_parsed_doc(
                        pg_conn, parsed, ner_extractor, entity_mapper
                    )
                except Exception as e:
                    logger.error("Failed to parse/map JATS XML at %s: %s", xml_path, e)
                    # Fail loud if needed, or propagate
                    raise e

            # Parse PDF files
            for pdf_path in pdf_paths:
                logger.info("Parsing PDF file: %s", pdf_path)
                try:
                    parsed = pdf_extractor.parse(pdf_path)
                    processed_docs_count += self._process_parsed_doc(
                        pg_conn, parsed, ner_extractor, entity_mapper
                    )
                except Exception as e:
                    logger.error("Failed to parse/map PDF at %s: %s", pdf_path, e)
                    raise e

            # Parse JSON metadata (abstract fallbacks) files
            for meta_path in metadata_paths:
                logger.info("Parsing abstract JSON file: %s", meta_path)
                try:
                    import json
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    text_chunks = meta.get("text_chunks")
                    if not text_chunks and "abstract" in meta and meta["abstract"]:
                        text_chunks = [{"section": "Abstract", "text": meta["abstract"]}]

                    if text_chunks:
                        parsed = {
                            "metadata": {
                                "title": meta.get("title"),
                                "journal": meta.get("journal"),
                                "doi": meta.get("doi"),
                                "pmid": meta.get("pmid"),
                                "pmcid": meta.get("pmcid"),
                                "published_date": meta.get("publication_date") or meta.get("pub_date"),
                            },
                            "text_chunks": text_chunks,
                            "references": meta.get("references") or []
                        }
                        processed_docs_count += self._process_parsed_doc(
                            pg_conn, parsed, ner_extractor, entity_mapper
                        )
                except Exception as e:
                    logger.error("Failed to parse/map abstract JSON at %s: %s", meta_path, e)
                    raise e

        finally:
            pg_conn.close()

        # Step 3: Index Chunks to LanceDB
        logger.info("Step 3: Index Chunks to LanceDB")
        # Re-open PG conn for steps that expect their own session or connections
        pg_conn_lancedb = psycopg.connect(self.pg_dsn)
        try:
            # Dynamically determine the embedding model's dimensions
            emb_client = EmbeddingClient()
            sample_emb = emb_client.embed_query("init_sample")
            vector_dim = len(sample_emb) if sample_emb else 384
            logger.info("Dynamically detected embedding dimension: %d", vector_dim)

            lancedb_client = LanceDBClient(uri=self.lancedb_uri, vector_dim=vector_dim)
            embeddings_pipeline = EmbeddingsPipeline(
                embedding_client=emb_client,
                vector_client=lancedb_client
            )
            indexed_chunks = embeddings_pipeline.index_pending_chunks(conn=pg_conn_lancedb)
        finally:
            pg_conn_lancedb.close()

        # Step 4: Materialize to Neo4j
        logger.info("Step 4: Materialize to Neo4j")
        pg_conn_neo4j = psycopg.connect(self.pg_dsn)
        neo4j_client = None
        try:
            neo4j_client = Neo4jClient(
                uri=self.neo4j_uri,
                user=self.neo4j_user,
                password=self.neo4j_password,
                database=self.neo4j_database
            )
            materializer = Materializer(neo4j_client=neo4j_client)
            # Sync approved/pending relationships (curation_statuses=None syncs all)
            materialized_stats = materializer.materialize_graph(
                pg_conn=pg_conn_neo4j,
                curation_statuses=["APPROVED", "PENDING"]
            )
        finally:
            if neo4j_client:
                neo4j_client.close()
            pg_conn_neo4j.close()


        return {
            "query": query,
            "scraped_count": scraper_res.get("total_found", 0),
            "processed_docs": processed_docs_count,
            "indexed_chunks": indexed_chunks,
            "materialized_stats": materialized_stats,
        }

    def _process_parsed_doc(
        self,
        pg_conn: Any,
        parsed: Dict[str, Any],
        ner_extractor: NERExtractor,
        entity_mapper: EntityMapper
    ) -> int:
        """Helper to extract entities and map document content into PostgreSQL."""
        text_chunks = parsed.get("text_chunks", [])
        if not text_chunks:
            logger.warning("Document parsed with no text chunks.")
            return 0

        ner_results = []
        for idx, chunk in enumerate(text_chunks):
            content = chunk.get("text") or chunk.get("content") or ""
            if not content.strip():
                continue

            entities = ner_extractor.extract(content)
            for ent in entities:
                ent["chunk_index"] = idx
                ner_results.append(ent)

        entity_mapper.map_document_content(
            conn=pg_conn,
            doc_metadata=parsed["metadata"],
            text_chunks=text_chunks,
            ner_results=ner_results
        )
        return 1
