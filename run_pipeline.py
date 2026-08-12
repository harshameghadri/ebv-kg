"""CLI entry point for running the unified ETL pipeline and initializing schemas."""

import argparse
import logging
import os
import sys

# Prevent multi-process thread thrashing across parallel worker slots
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
try:
    import torch
    torch.set_num_threads(2)
except Exception:
    pass


import psycopg


from app.database.schema import init_db_schema
from app.retrieval.vector import LanceDBClient
from app.materialization.neo4j_client import Neo4jClient
from app.materialization.materializer import Materializer
from app.ingestion.pipeline import ETLPipeline

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EBV Knowledge System: Unified ETL Pipeline Orchestrator"
    )
    parser.add_argument(
        "--query", "-q",
        required=True,
        help="PubMed search query."
    )
    parser.add_argument(
        "--max-articles", "-m",
        type=int,
        default=5,
        help="Maximum number of articles to download/process (default: 5)."
    )
    parser.add_argument(
        "--staging-dir",
        default="data/staging",
        help="Base directory for staging downloads (default: data/staging)."
    )
    parser.add_argument(
        "--pg-dsn",
        default=os.getenv("DATABASE_URL") or os.getenv("POSTGRES_DSN"),
        help="PostgreSQL DSN."
    )
    parser.add_argument(
        "--lancedb-uri",
        default=os.getenv("LANCEDB_URI", "data/lancedb/"),
        help="LanceDB database URI."
    )
    parser.add_argument(
        "--neo4j-uri",
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j connection URI."
    )
    parser.add_argument(
        "--neo4j-user",
        default=os.getenv("NEO4J_USER", "neo4j"),
        help="Neo4j username."
    )
    parser.add_argument(
        "--neo4j-password",
        default=os.getenv("NEO4J_PASSWORD", "password"),
        help="Neo4j password."
    )
    parser.add_argument(
        "--neo4j-database",
        default=os.getenv("NEO4J_DATABASE", "neo4j"),
        help="Neo4j database name."
    )
    parser.add_argument(
        "--skip-init",
        action="store_true",
        help="Skip database schema/index initialization."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging."
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not args.pg_dsn:
        logger.error(
            "PostgreSQL connection DSN must be provided via --pg-dsn or environment variable DATABASE_URL/POSTGRES_DSN."
        )
        sys.exit(1)

    # 1. Initialize Schemas
    if not args.skip_init:
        logger.info("Initializing PostgreSQL schema...")
        try:
            with psycopg.connect(args.pg_dsn) as conn:
                init_db_schema(conn)
            logger.info("PostgreSQL schema initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize PostgreSQL schema: %s", e)
            sys.exit(1)

        logger.info("Initializing LanceDB table...")
        try:
            lancedb_client = LanceDBClient(uri=args.lancedb_uri)
            lancedb_client.init_table()
            logger.info("LanceDB table initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize LanceDB table: %s", e)
            sys.exit(1)

        logger.info("Initializing Neo4j indices and constraints...")
        try:
            neo4j_client = Neo4jClient(
                uri=args.neo4j_uri,
                user=args.neo4j_user,
                password=args.neo4j_password,
                database=args.neo4j_database
            )
            materializer = Materializer(neo4j_client=neo4j_client)
            materializer.init_schema()
            neo4j_client.close()
            logger.info("Neo4j schema initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize Neo4j schema: %s", e)
            sys.exit(1)

    # 2. Run Pipeline
    logger.info("Instantiating ETLPipeline...")
    pipeline = ETLPipeline(
        pg_dsn=args.pg_dsn,
        lancedb_uri=args.lancedb_uri,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database,
        staging_dir=args.staging_dir
    )

    logger.info("Executing run_etl_pipeline...")
    try:
        stats = pipeline.run_etl_pipeline(args.query, max_articles=args.max_articles)
        logger.info("Pipeline execution completed successfully.")
        print("Pipeline Execution Summary:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    except Exception as e:
        logger.error("Pipeline execution failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
