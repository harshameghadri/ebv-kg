"""
Unit tests for AnnData CLI command in app/ingestion/anndata_cli.py
"""

import json
import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch

import pandas as pd

from app.ingestion.anndata_cli import build_parser, run_anndata_cli, main


@pytest.fixture
def sample_marker_csv():
    """Creates a temporary marker CSV file for testing."""
    df = pd.DataFrame({
        "gene": ["TBX21", "CXCR3", "CD19"],
        "cluster": ["Atypical B Cell", "Atypical B Cell", "Memory B Cell"],
        "avg_log2FC": [2.5, 1.8, 0.9],
        "p_val_adj": [1e-5, 1e-4, 1e-2]
    })
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def sample_marker_tsv():
    """Creates a temporary marker TSV file for testing."""
    df = pd.DataFrame({
        "gene_symbol": ["EBNA1", "LMP1"],
        "cell_type": ["Latent B Cell", "Latent B Cell"],
        "log2fc": [3.1, 2.7],
        "padj": [1e-6, 1e-5]
    })
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        df.to_csv(f.name, sep="\t", index=False)
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.remove(temp_path)


def test_arg_parser_defaults():
    """Tests default values in build_parser."""
    parser = build_parser()
    args = parser.parse_args(["--input", "test.csv"])
    assert args.input == "test.csv"
    assert args.cluster_key == "cell_type"
    assert args.source_id == "single_cell_dataset"
    assert args.verbose is False


def test_run_cli_marker_csv_no_db(sample_marker_csv):
    """Tests CLI execution on marker CSV without DB DSN."""
    output = run_anndata_cli(["--input", sample_marker_csv])
    assert output["status"] == "success"
    assert output["format"] == "marker_csv"
    assert output["records_parsed"] == 3
    assert output["db_saved"] is None


def test_run_cli_marker_tsv(sample_marker_tsv):
    """Tests CLI execution on TSV marker file."""
    output = run_anndata_cli(["--input", sample_marker_tsv, "--source-id", "ebv_study_1"])
    assert output["status"] == "success"
    assert output["format"] == "marker_csv"
    assert output["records_parsed"] == 2


@patch("psycopg.connect")
def test_run_cli_marker_csv_with_db(mock_connect, sample_marker_csv):
    """Tests CLI execution on marker CSV with PostgreSQL connection mock."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.rowcount = 1
    mock_conn.cursor.return_value = mock_cur
    mock_connect.return_value = mock_conn

    output = run_anndata_cli([
        "--input", sample_marker_csv,
        "--pg-dsn", "postgres://user:pass@localhost:5432/testdb"
    ])

    assert output["status"] == "success"
    assert output["db_saved"] is not None
    assert output["db_saved"]["inserted_entities"] >= 0
    assert output["db_saved"]["inserted_relationships"] >= 0
    mock_conn.commit.assert_called_once()
    mock_conn.close.assert_called_once()


@patch("app.ingestion.anndata_parser.AnnDataParser.parse_h5ad_file")
def test_run_cli_h5ad_no_db(mock_parse_h5ad):
    """Tests CLI execution on .h5ad file without DB connection."""
    mock_parse_h5ad.return_value = {
        "n_cells": 1500,
        "n_genes": 20000,
        "cluster_key": "cell_type",
        "cell_types": ["Atypical B Cell", "Plasmablast"],
        "file_path": "sample.h5ad"
    }

    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
        h5ad_path = f.name

    try:
        output = run_anndata_cli(["--input", h5ad_path, "--cluster-key", "cell_type"])
        assert output["status"] == "success"
        assert output["format"] == "h5ad"
        assert output["summary"]["n_cells"] == 1500
        assert len(output["summary"]["cell_types"]) == 2
    finally:
        if os.path.exists(h5ad_path):
            os.remove(h5ad_path)


@patch("psycopg.connect")
@patch("app.ingestion.anndata_parser.AnnDataParser.parse_h5ad_file")
def test_run_cli_h5ad_with_db(mock_parse_h5ad, mock_connect):
    """Tests CLI execution on .h5ad file with DB cell type upsert."""
    mock_parse_h5ad.return_value = {
        "n_cells": 1000,
        "n_genes": 18000,
        "cluster_key": "cell_type",
        "cell_types": ["Atypical B Cell", "GCB"],
        "file_path": "sample.h5ad"
    }
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.rowcount = 1
    mock_conn.cursor.return_value = mock_cur
    mock_connect.return_value = mock_conn

    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
        h5ad_path = f.name

    try:
        output = run_anndata_cli([
            "--input", h5ad_path,
            "--pg-dsn", "postgres://user:pass@localhost:5432/testdb"
        ])

        assert output["status"] == "success"
        assert output["db_saved"] == {"inserted_cell_types": 2}
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()
    finally:
        if os.path.exists(h5ad_path):
            os.remove(h5ad_path)


def test_main_file_not_found(capsys):
    """Tests main entry point error handling when file does not exist."""
    exit_code = main(["--input", "/non/existent/file.csv"])
    assert exit_code == 1

    captured = capsys.readouterr()
    error_data = json.loads(captured.err)
    assert error_data["status"] == "error"
    assert "not found" in error_data["error"].lower()


def test_main_invalid_csv_columns(capsys):
    """Tests main error handling when marker CSV is missing required columns."""
    df = pd.DataFrame({"invalid_column": [1, 2, 3]})
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        invalid_path = f.name

    try:
        exit_code = main(["--input", invalid_path])
        assert exit_code == 1

        captured = capsys.readouterr()
        error_data = json.loads(captured.err)
        assert error_data["status"] == "error"
        assert "missing required columns" in error_data["error"].lower()
    finally:
        if os.path.exists(invalid_path):
            os.remove(invalid_path)


@patch("psycopg.connect")
def test_main_db_connection_failure(mock_connect, sample_marker_csv, capsys):
    """Tests main error handling when PostgreSQL connection fails."""
    mock_connect.side_effect = Exception("Connection timeout")

    exit_code = main([
        "--input", sample_marker_csv,
        "--pg-dsn", "postgres://invalid:pass@localhost:5432/db"
    ])
    assert exit_code == 1

    captured = capsys.readouterr()
    error_data = json.loads(captured.err)
    assert error_data["status"] == "error"
    assert "connection error" in error_data["error"].lower()


def test_main_success_stdout(sample_marker_csv, capsys):
    """Tests main success printing valid JSON output to stdout."""
    exit_code = main(["--input", sample_marker_csv])
    assert exit_code == 0

    captured = capsys.readouterr()
    output_data = json.loads(captured.out)
    assert output_data["status"] == "success"
    assert output_data["format"] == "marker_csv"
    assert output_data["records_parsed"] == 3
