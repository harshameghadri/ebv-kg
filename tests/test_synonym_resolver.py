"""Tests for local dictionary-based synonym resolver with OLS fallback."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from app.processing.synonym_resolver import SynonymResolver, normalize_term


def test_normalize_term():
    assert normalize_term("  EBNA-1  ") == "ebna 1"
    assert normalize_term("B-cell (CD4+)") == "b cell cd4"
    assert normalize_term("") == ""


def test_exact_match_local_dict():
    dict_data = {
        "HGNC": {
            "EBNA1": {
                "canonical_id": "HGNC:3236",
                "symbol": "EBNA1",
                "aliases": ["EBNA-1"],
            },
            "CXCR3": "HGNC:15525",
        },
        "CELL_TYPE": {
            "B cell": {
                "canonical_id": "CL:0000236",
                "symbol": "B cell",
                "aliases": ["B-cell", "B lymphocyte"],
            },
        },
        "DISEASE": {
            "Infectious mononucleosis": "DOID:8569",
        },
        "PROTEIN": {
            "LMP1": {"canonical_id": "P03230", "symbol": "LMP1"},
        },
        "UBERON": {
            "spleen": "UBERON:0002106",
        },
    }

    resolver = SynonymResolver(dictionaries=dict_data, ols_enabled=False)

    # 1. Exact raw match for HGNC
    res = resolver.resolve("EBNA1", category="HGNC")
    assert res is not None
    assert res["canonical_id"] == "HGNC:3236"
    assert res["symbol"] == "EBNA1"
    assert res["match_type"] == "exact"
    assert res["confidence"] == 1.0

    # 2. Exact match via alias
    res_alias = resolver.resolve("EBNA-1", category="GENE")
    assert res_alias is not None
    assert res_alias["canonical_id"] == "HGNC:3236"

    # 3. Exact match for Cell Ontology
    res_cl = resolver.resolve("B cell", category="CELL_TYPE")
    assert res_cl is not None
    assert res_cl["canonical_id"] == "CL:0000236"

    # 4. Exact match for Disease Ontology
    res_doid = resolver.resolve("Infectious mononucleosis", category="DISEASE")
    assert res_doid is not None
    assert res_doid["canonical_id"] == "DOID:8569"

    # 5. Exact match for UniProt
    res_uniprot = resolver.resolve("LMP1", category="PROTEIN")
    assert res_uniprot is not None
    assert res_uniprot["canonical_id"] == "P03230"

    # 6. Exact match for UBERON
    res_uberon = resolver.resolve("spleen", category="UBERON")
    assert res_uberon is not None
    assert res_uberon["canonical_id"] == "UBERON:0002106"


def test_normalized_and_fuzzy_match():
    dict_data = {
        "DISEASE": [
            {
                "term": "Epstein-Barr virus infectious disease",
                "canonical_id": "DOID:0050741",
                "symbol": "Epstein-Barr virus infectious disease",
                "aliases": ["EBV infection"],
            }
        ]
    }

    resolver = SynonymResolver(
        dictionaries=dict_data, ols_enabled=False, fuzzy_threshold=0.75
    )

    # Normalized exact match (ignores hyphens / extra spaces)
    res_norm = resolver.resolve(
        "Epstein Barr virus infectious disease", category="DISEASE"
    )
    assert res_norm is not None
    assert res_norm["canonical_id"] == "DOID:0050741"
    assert res_norm["match_type"] == "exact"

    # Fuzzy match with slight typo
    res_fuzzy = resolver.resolve(
        "Epstein-Bar virus infectious disease", category="DISEASE"
    )
    assert res_fuzzy is not None
    assert res_fuzzy["canonical_id"] == "DOID:0050741"
    assert res_fuzzy["match_type"] == "fuzzy"
    assert res_fuzzy["confidence"] >= 0.75

    # Below fuzzy threshold
    res_none = resolver.resolve("completely unrelated illness", category="DISEASE")
    assert res_none is None


def test_loading_from_csv_and_json_files(tmp_path: Path):
    csv_file = tmp_path / "hgnc.csv"
    csv_file.write_text(
        "term,canonical_id,symbol,aliases\nEBNA2,HGNC:3237,EBNA2,EBNA-2;BYRF1\n",
        encoding="utf-8",
    )

    json_file = tmp_path / "cl.json"
    json_data = [
        {
            "term": "T cell",
            "canonical_id": "CL:0000084",
            "symbol": "T cell",
            "aliases": ["T-cell"],
        }
    ]
    json_file.write_text(json.dumps(json_data), encoding="utf-8")

    resolver = SynonymResolver(dictionary_dir=tmp_path, ols_enabled=False)

    res_csv = resolver.resolve("EBNA-2", category="HGNC")
    assert res_csv is not None
    assert res_csv["canonical_id"] == "HGNC:3237"

    res_json = resolver.resolve("T-cell", category="CELL_TYPE")
    assert res_json is not None
    assert res_json["canonical_id"] == "CL:0000084"


def test_ols_fallback_success():
    resolver = SynonymResolver(ols_enabled=True, ols_timeout=2.0)

    mock_ols_response = {
        "response": {
            "docs": [
                {
                    "obo_id": "DOID:0050741",
                    "label": "Epstein-Barr virus infectious disease",
                    "ontology_prefix": "DOID",
                }
            ]
        }
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_ols_response

    with patch("requests.get", return_value=mock_response) as mock_get:
        res = resolver.resolve(
            "Epstein-Barr virus infectious disease", category="DISEASE"
        )
        assert res is not None
        assert res["canonical_id"] == "DOID:0050741"
        assert res["symbol"] == "Epstein-Barr virus infectious disease"
        assert res["match_type"] == "ols"
        assert res["source"] == "ols_doid"
        assert res["confidence"] == 0.95

        mock_get.assert_called_once_with(
            "https://www.ebi.ac.uk/ols4/api/search",
            params={
                "q": "Epstein-Barr virus infectious disease",
                "rows": 5,
                "ontology": "doid",
            },
            timeout=2.0,
        )


def test_ols_fallback_failure_and_timeout():
    resolver = SynonymResolver(ols_enabled=True)

    with patch(
        "requests.get", side_effect=requests.RequestException("Connection timeout")
    ):
        res = resolver.resolve("UnknownEntity123", category="GENE")
        assert res is None


def test_invalid_category_and_missing_file():
    resolver = SynonymResolver()
    res = resolver.resolve("some term", category="INVALID_CAT")
    assert res is None

    with pytest.raises(FileNotFoundError):
        resolver.load_dictionary("HGNC", "/nonexistent/path/hgnc.csv")

