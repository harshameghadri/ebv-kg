"""
CLI Command for Single-Cell AnnData (.h5ad) and Marker CSV Ingestion

Parses .h5ad files or marker gene CSVs using AnnDataParser,
saves normalized entities and relationships to PostgreSQL (if DSN provided),
and outputs a JSON execution summary.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Any, Optional
import pandas as pd

from app.ingestion.anndata_parser import AnnDataParser

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Builds and returns the argparse ArgumentParser for AnnData CLI."""
    parser = argparse.ArgumentParser(
        description="Ingest single-cell AnnData (.h5ad) files or marker gene CSVs into PostgreSQL EBV KG."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to single-cell AnnData (.h5ad) file or marker gene CSV/TSV file."
    )
    parser.add_argument(
        "--cluster-key",
        default="cell_type",
        help="Cluster annotation key in AnnData obs metadata (default: cell_type)."
    )
    parser.add_argument(
        "--pg-dsn",
        default=os.getenv("DATABASE_URL") or os.getenv("POSTGRES_DSN"),
        help="PostgreSQL connection DSN string."
    )
    parser.add_argument(
        "--source-id",
        default="single_cell_dataset",
        help="Dataset source identifier for relationship provenance (default: single_cell_dataset)."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debugging logs."
    )
    return parser


def run_anndata_cli(args_list: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Executes the AnnData CLI logic.
    Accepts optional list of command-line arguments (for unit testing).
    Returns a result dictionary.
    """
    parser = build_parser()
    args = parser.parse_args(args_list)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    input_path = args.input
    cluster_key = args.cluster_key
    pg_dsn = args.pg_dsn
    source_id = args.source_id

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    conn = None
    if pg_dsn:
        try:
            import psycopg
            conn = psycopg.connect(pg_dsn)
            logger.info("Successfully connected to PostgreSQL database.")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL database: {e}")
            raise RuntimeError(f"Database connection error: {e}") from e

    anndata_parser = AnnDataParser(db_connection=conn)

    try:
        if input_path.lower().endswith(".h5ad"):
            summary = anndata_parser.parse_h5ad_file(input_path, cluster_key=cluster_key)
            db_saved = None

            if conn and summary.get("cell_types"):
                cur = conn.cursor()
                inserted_ct = 0
                for ct in summary["cell_types"]:
                    cur.execute(
                        """
                        INSERT INTO normalized_entities (canonical_id, name, entity_type, ontology_source)
                        VALUES (%s, %s, 'CELL_TYPE', 'CL')
                        ON CONFLICT (canonical_id) DO NOTHING;
                        """,
                        (f"CL:{str(ct).replace(' ', '_')}", str(ct))
                    )
                    inserted_ct += cur.rowcount
                conn.commit()
                db_saved = {"inserted_cell_types": inserted_ct}


            output = {
                "status": "success",
                "input_file": input_path,
                "format": "h5ad",
                "summary": summary,
                "db_saved": db_saved
            }
        else:
            sep = "\t" if input_path.lower().endswith((".tsv", ".txt")) else ","
            try:
                df = pd.read_csv(input_path, sep=sep)
            except Exception:
                df = pd.read_csv(input_path, sep=None, engine="python")

            records = anndata_parser.parse_marker_dataframe(df, source_id=source_id)
            db_saved = None

            if conn:
                db_saved = anndata_parser.save_markers_to_db(records)

            output = {
                "status": "success",
                "input_file": input_path,
                "format": "marker_csv",
                "records_parsed": len(records),
                "db_saved": db_saved
            }

        return output

    finally:
        if conn:
            conn.close()
            logger.info("PostgreSQL database connection closed.")


def main(args_list: Optional[List[str]] = None) -> int:
    """Main CLI entry point with JSON output printing and clean exception handling."""
    try:
        result = run_anndata_cli(args_list)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as e:
        error_output = {
            "status": "error",
            "error": str(e)
        }
        print(json.dumps(error_output, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
