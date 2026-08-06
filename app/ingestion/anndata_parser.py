"""
AnnData (.h5ad) and Single-Cell Cluster Marker Parser

Ingests single-cell RNA-seq dataset files (.h5ad) or exported cluster marker CSVs (Scanpy/Seurat format).
Maps cluster marker genes to cell state entities (e.g., Atypical B Cell, GCB, Plasmablast)
and writes normalized CellState nodes and IS_MARKER_FOR relationships to PostgreSQL.
"""

import os
import logging
import pandas as pd
from typing import Dict, List, Any, Optional

try:
    import anndata as ad
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False

logger = logging.getLogger(__name__)

class AnnDataParser:
    """Parses single-cell RNA-seq marker datasets and AnnData objects."""

    def __init__(self, db_connection=None):
        self.conn = db_connection

    def parse_marker_dataframe(self, df: pd.DataFrame, source_id: str = "single_cell_dataset") -> List[Dict[str, Any]]:
        """
        Parses a marker genes DataFrame (Scanpy/Seurat format).
        Expected columns: gene (or names), cluster (or cell_type), logfoldchanges (or avg_log2FC), pvals_adj (or p_val_adj).
        """
        records = []
        
        # Standardize column names
        col_map = {}
        for col in df.columns:
            lower = col.lower()
            if lower in ["gene", "names", "gene_symbol", "feature"]:
                col_map[col] = "gene"
            elif lower in ["cluster", "cell_type", "celltype", "group"]:
                col_map[col] = "cluster"
            elif lower in ["logfoldchanges", "avg_log2fc", "log2fc", "avg_logfc"]:
                col_map[col] = "log2fc"
            elif lower in ["pvals_adj", "p_val_adj", "padj", "adj_pval"]:
                col_map[col] = "p_val_adj"

        df_std = df.rename(columns=col_map)
        
        if "gene" not in df_std.columns or "cluster" not in df_std.columns:
            raise ValueError(f"Marker DataFrame missing required columns ('gene', 'cluster'). Found: {list(df.columns)}")

        for _, row in df_std.iterrows():
            gene = str(row["gene"]).strip()
            cluster = str(row["cluster"]).strip()
            log2fc = float(row["log2fc"]) if "log2fc" in df_std.columns and pd.notna(row["log2fc"]) else 1.0
            p_adj = float(row["p_val_adj"]) if "p_val_adj" in df_std.columns and pd.notna(row["p_val_adj"]) else 0.01

            # Compute relationship confidence (higher log2fc and lower p_adj -> higher confidence)
            confidence = min(0.98, max(0.50, 0.70 + (log2fc / 10.0)))

            record = {
                "gene_symbol": gene,
                "cell_state": cluster,
                "relationship_type": "IS_MARKER_FOR",
                "confidence": round(confidence, 3),
                "log2fc": round(log2fc, 3),
                "p_val_adj": p_adj,
                "source_id": source_id
            }
            records.append(record)

        logger.info(f"Parsed {len(records)} marker gene relationships from single-cell DataFrame.")
        return records

    def parse_h5ad_file(self, h5ad_path: str, cluster_key: str = "cell_type") -> Dict[str, Any]:
        """Reads an AnnData .h5ad file and extracts summary cell state metadata."""
        if not HAS_ANNDATA:
            raise RuntimeError("anndata package is required to parse .h5ad files.")

        if not os.path.exists(h5ad_path):
            raise FileNotFoundError(f"AnnData file not found: {h5ad_path}")

        adata = ad.read_h5ad(h5ad_path, backed="r")
        n_obs, n_vars = adata.n_obs, adata.n_vars
        
        cell_types = []
        if cluster_key in adata.obs.columns:
            cell_types = list(adata.obs[cluster_key].unique())

        return {
            "n_cells": n_obs,
            "n_genes": n_vars,
            "cluster_key": cluster_key,
            "cell_types": cell_types,
            "file_path": h5ad_path
        }

    def save_markers_to_db(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        """Inserts parsed marker relationships and normalized entities into PostgreSQL."""
        if not self.conn:
            logger.warning("No DB connection provided. Returning parsed record counts.")
            return {"inserted_relationships": len(records), "inserted_entities": 0}

        cur = self.conn.cursor()
        entities_inserted = 0
        relationships_inserted = 0

        for r in records:
            gene = r["gene_symbol"]
            cell_state = r["cell_state"]
            conf = r["confidence"]

            # 1. Upsert Gene entity
            cur.execute("""
                INSERT INTO normalized_entities (canonical_id, name, entity_type, ontology_source)
                VALUES (%s, %s, 'GENE', 'HGNC')
                ON CONFLICT (canonical_id) DO NOTHING;
            """, (f"HGNC:{gene}", gene))
            entities_inserted += cur.rowcount

            # 2. Upsert CellState entity
            cur.execute("""
                INSERT INTO normalized_entities (canonical_id, name, entity_type, ontology_source)
                VALUES (%s, %s, 'CELL_TYPE', 'CL')
                ON CONFLICT (canonical_id) DO NOTHING;
            """, (f"CL:{cell_state.replace(' ', '_')}", cell_state))
            entities_inserted += cur.rowcount

            # 3. Upsert IS_MARKER_FOR relationship
            status = "APPROVED" if conf >= 0.80 else "PENDING"
            cur.execute("""
                INSERT INTO relationships (source_entity_id, target_entity_id, relationship_type, confidence, curation_status)
                VALUES (%s, %s, 'IS_MARKER_FOR', %s, %s)
                ON CONFLICT DO NOTHING;
            """, (f"HGNC:{gene}", f"CL:{cell_state.replace(' ', '_')}", conf, status))
            relationships_inserted += cur.rowcount

        self.conn.commit()

        return {
            "inserted_entities": entities_inserted,
            "inserted_relationships": relationships_inserted
        }
