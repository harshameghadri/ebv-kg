import argparse
import logging
import os
import sys
import psycopg
from app.materialization.neo4j_client import Neo4jClient
from app.discovery.lightrag_runner import DiscoveryClusteringRunner
from app.discovery.harvest import DiscoveryHarvester

logger = logging.getLogger("app.discovery.cli")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated concept clustering and candidate harvesting CLI"
    )
    parser.add_argument(
        "--pg-dsn",
        default=os.getenv("DATABASE_URL") or os.getenv("POSTGRES_DSN"),
        help="PostgreSQL connection DSN"
    )
    parser.add_argument(
        "--neo4j-uri",
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j connection URI"
    )
    parser.add_argument(
        "--neo4j-user",
        default=os.getenv("NEO4J_USER", "neo4j"),
        help="Neo4j username"
    )
    parser.add_argument(
        "--neo4j-password",
        default=os.getenv("NEO4J_PASSWORD", "password"),
        help="Neo4j password"
    )
    parser.add_argument(
        "--neo4j-database",
        default=os.getenv("NEO4J_DATABASE", "neo4j"),
        help="Neo4j database name"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum confidence threshold for clustering"
    )
    parser.add_argument(
        "--algorithm",
        default="label_propagation",
        choices=["label_propagation", "connected_components"],
        help="Clustering algorithm to use"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=20,
        help="Max iterations for label propagation"
    )
    parser.add_argument(
        "--export-json",
        default=None,
        help="Optional path to export candidates JSON file"
    )
    parser.add_argument(
        "--insert-db",
        action="store_true",
        default=True,
        help="Insert candidates into the curation queue"
    )
    parser.add_argument(
        "--no-insert-db",
        action="store_false",
        dest="insert_db",
        help="Do not insert candidates into the curation queue"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Limit on candidates harvested"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debugging logs"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if not args.pg_dsn:
        logger.error("PostgreSQL DSN must be provided via --pg-dsn or DATABASE_URL/POSTGRES_DSN env vars.")
        sys.exit(1)

    logger.info("Connecting to PostgreSQL...")
    try:
        pg_conn = psycopg.connect(args.pg_dsn)
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        sys.exit(1)

    logger.info("Connecting to Neo4j...")
    neo4j_client = None
    try:
        neo4j_client = Neo4jClient(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password,
            database=args.neo4j_database
        )
    except Exception as e:
        logger.warning(f"Failed to connect to Neo4j, using mocked/empty Neo4j connection: {e}")

    try:
        logger.info("Running clustering runner...")
        runner = DiscoveryClusteringRunner(pg_conn=pg_conn)
        communities = runner.run_clustering(
            min_confidence=args.min_confidence,
            algorithm=args.algorithm,
            max_iter=args.max_iter
        )
        logger.info(f"Detected {len(communities)} communities.")

        logger.info("Harvesting candidate links...")
        harvester = DiscoveryHarvester(pg_conn=pg_conn, neo4j_client=neo4j_client)
        candidates = harvester.harvest_candidates(
            communities=communities,
            export_json_path=args.export_json,
            insert_to_db=args.insert_db,
            limit=args.limit
        )
        logger.info(f"Successfully harvested {len(candidates)} candidates.")

    except Exception as e:
        logger.error(f"Error during discovery execution: {e}")
        sys.exit(1)
    finally:
        pg_conn.close()
        if neo4j_client:
            neo4j_client.close()

if __name__ == "__main__":
    main()
