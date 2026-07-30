"""Unit tests for GEOCrawler and GEO metadata parser functions."""

import json
from unittest.mock import MagicMock

import pytest
import requests

from app.ingestion.geo_crawler import (
    GEOCrawler,
    parse_sample_characteristics,
    parse_series_matrix_header,
)

MOCK_ESEARCH_JSON = {
    "esearchresult": {
        "count": "1",
        "retmax": "1",
        "retstart": "0",
        "idlist": ["200189141"],
    }
}

MOCK_ESEARCH_EMPTY_JSON = {
    "esearchresult": {
        "count": "0",
        "idlist": [],
    }
}

MOCK_ESUMMARY_JSON = {
    "result": {
        "200189141": {
            "title": "Single-cell RNA sequencing of EBV infected B cells",
            "summary": "Single-cell profiling of host and viral transcriptomes.",
            "gdsType": "Expression profiling by high throughput sequencing",
            "gse": "189141",
            "organism": "Homo sapiens",
            "overall_design": "Single cell analysis across multiple donors.",
            "samples": [
                {"accession": "GSM5696078", "title": "PBMC_Donor1"},
                {"accession": "GSM5696079", "title": "PBMC_Donor2"},
            ],
        }
    }
}

MOCK_SERIES_MATRIX_TEXT = """^SERIES = GSE189141
!Series_title = "Single-cell RNA sequencing of EBV infected B cells"
!Series_summary = "Single-cell profiling of host and viral transcriptomes."
!Series_overall_design = "4 single-cell libraries prepared using 10x Genomics."
!Series_type = "Expression profiling by high throughput sequencing"
!Series_sample_organism = "Homo sapiens"
!Sample_title = "PBMC_Donor1"	"PBMC_Donor2"
!Sample_geo_accession = "GSM5696078"	"GSM5696079"
!Sample_characteristics_ch1 = "cell type: B cell"	"cell type: CD8+ T cell"
!Sample_characteristics_ch1 = "disease: IM"	"disease: Healthy"
!Sample_characteristics_ch1 = "tissue: blood"	"tissue: blood"
!Sample_source_name_ch1 = "Blood"	"Blood"
!series_matrix_table_begin
ID_REF	GSM5696078	GSM5696079
"""


def test_parse_sample_characteristics_basic():
    raw_chars = [
        "cell type: B cell",
        "disease state: Infectious Mononucleosis",
        "tissue source: peripheral blood",
        "donor: Donor_A",
    ]
    res = parse_sample_characteristics(raw_chars)
    assert res["cell_type"] == "B cell"
    assert res["disease_state"] == "Infectious Mononucleosis"
    assert res["tissue_source"] == "peripheral blood"
    assert res["attributes"]["cell type"] == "B cell"
    assert res["attributes"]["donor"] == "Donor_A"
    assert res["raw_characteristics"] == raw_chars


def test_parse_sample_characteristics_variations():
    raw_chars = [
        "cell_line: Akata",
        "condition: Burkitt Lymphoma",
        "organ: Lymph node",
    ]
    res = parse_sample_characteristics(raw_chars)
    assert res["cell_type"] == "Akata"
    assert res["disease_state"] == "Burkitt Lymphoma"
    assert res["tissue_source"] == "Lymph node"


def test_parse_series_matrix_header():
    res = parse_series_matrix_header(MOCK_SERIES_MATRIX_TEXT)
    assert "Single-cell RNA sequencing" in res["title"]
    assert "Single-cell profiling" in res["summary"]
    assert res["gds_type"] == "Expression profiling by high throughput sequencing"
    assert res["organism"] == "Homo sapiens"
    assert len(res["samples"]) == 2

    s0 = res["samples"][0]
    assert s0["accession"] == "GSM5696078"
    assert s0["title"] == "PBMC_Donor1"
    assert s0["cell_type"] == "B cell"
    assert s0["disease_state"] == "IM"
    assert s0["tissue_source"] == "blood"

    s1 = res["samples"][1]
    assert s1["accession"] == "GSM5696079"
    assert s1["title"] == "PBMC_Donor2"
    assert s1["cell_type"] == "CD8+ T cell"
    assert s1["disease_state"] == "Healthy"


def test_search_gse_invalid_id():
    crawler = GEOCrawler()
    with pytest.raises(ValueError, match="Invalid GSE series ID format"):
        crawler.search_gse("INVALID_123")
    with pytest.raises(ValueError, match="Invalid GSE series ID format"):
        crawler.search_gse("")


def test_search_gse_not_found(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = MOCK_ESEARCH_EMPTY_JSON
    mock_resp.raise_for_status.return_value = None

    session = requests.Session()
    monkeypatch.setattr(session, "get", lambda *args, **kwargs: mock_resp)

    crawler = GEOCrawler(session=session)
    with pytest.raises(ValueError, match="No NCBI GDS dataset found"):
        crawler.search_gse("GSE99999999")


def test_search_gse_success(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = MOCK_ESEARCH_JSON
    mock_resp.raise_for_status.return_value = None

    session = requests.Session()
    monkeypatch.setattr(session, "get", lambda *args, **kwargs: mock_resp)

    crawler = GEOCrawler(session=session)
    uid = crawler.search_gse("GSE189141")
    assert uid == "200189141"


def test_fetch_summary_success(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = MOCK_ESUMMARY_JSON
    mock_resp.raise_for_status.return_value = None

    session = requests.Session()
    monkeypatch.setattr(session, "get", lambda *args, **kwargs: mock_resp)

    crawler = GEOCrawler(session=session)
    summary = crawler.fetch_summary("200189141")
    assert "Single-cell RNA sequencing" in summary["title"]
    assert summary["gse"] == "189141"


def test_fetch_summary_not_found(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"result": {}}
    mock_resp.raise_for_status.return_value = None

    session = requests.Session()
    monkeypatch.setattr(session, "get", lambda *args, **kwargs: mock_resp)

    crawler = GEOCrawler(session=session)
    with pytest.raises(
        ValueError, match="Summary record for UID '200189141' not found"
    ):
        crawler.fetch_summary("200189141")


def test_fetch_gse_and_crawl_and_stage(tmp_path, monkeypatch):
    def mock_get(url, params=None, timeout=None):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        if "esearch.fcgi" in url:
            mock_resp.json.return_value = MOCK_ESEARCH_JSON
        elif "esummary.fcgi" in url:
            mock_resp.json.return_value = MOCK_ESUMMARY_JSON
        elif "acc.cgi" in url:
            mock_resp.text = MOCK_SERIES_MATRIX_TEXT
        else:
            raise RuntimeError(f"Unexpected URL: {url}")
        return mock_resp

    session = requests.Session()
    monkeypatch.setattr(session, "get", mock_get)

    staging_dir = tmp_path / "staging" / "geo"
    crawler = GEOCrawler(output_dir=staging_dir, session=session)

    out_file = crawler.crawl_and_stage("GSE189141")
    assert out_file.exists()
    assert out_file.name == "GSE189141.json"

    with out_file.open("r", encoding="utf-8") as f:
        staged_data = json.load(f)

    assert staged_data["gse_id"] == "GSE189141"
    assert staged_data["uid"] == "200189141"
    assert len(staged_data["samples"]) == 2
    assert staged_data["samples"][0]["accession"] == "GSM5696078"
    assert staged_data["samples"][0]["cell_type"] == "B cell"
    assert staged_data["samples"][0]["disease_state"] == "IM"
    assert staged_data["samples"][0]["tissue_source"] == "blood"


def test_http_failure_raises_runtime_error(monkeypatch):
    session = requests.Session()

    def mock_get_fail(url, params=None, timeout=None):
        raise requests.RequestException("Connection timeout")

    monkeypatch.setattr(session, "get", mock_get_fail)

    crawler = GEOCrawler(session=session)
    with pytest.raises(RuntimeError, match="Entrez esearch request failed"):
        crawler.search_gse("GSE189141")
