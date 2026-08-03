import pytest
import pandas as pd
from app.ingestion.anndata_parser import AnnDataParser

def test_parse_marker_dataframe():
    parser = AnnDataParser()
    df = pd.DataFrame({
        "gene": ["TBX21", "CXCR3", "CD19"],
        "cluster": ["Atypical B Cell", "Atypical B Cell", "Memory B Cell"],
        "avg_log2FC": [2.5, 1.8, 0.9],
        "p_val_adj": [1e-5, 1e-4, 1e-2]
    })

    records = parser.parse_marker_dataframe(df)
    assert len(records) == 3
    assert records[0]["gene_symbol"] == "TBX21"
    assert records[0]["cell_state"] == "Atypical B Cell"
    assert records[0]["relationship_type"] == "IS_MARKER_FOR"
    assert records[0]["confidence"] >= 0.80

def test_parse_marker_dataframe_missing_cols():
    parser = AnnDataParser()
    df = pd.DataFrame({"invalid_col": [1, 2]})
    with pytest.raises(ValueError, match="missing required columns"):
        parser.parse_marker_dataframe(df)
