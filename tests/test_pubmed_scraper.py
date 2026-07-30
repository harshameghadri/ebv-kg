"""Unit tests for PubMedScraper."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from app.ingestion.pubmed_scraper import PubMedScraper, main


@pytest.fixture
def temp_staging(tmp_path: Path) -> Path:
    return tmp_path / "staging"


@pytest.fixture
def scraper(temp_staging: Path) -> PubMedScraper:
    return PubMedScraper(
        staging_dir=temp_staging,
        email="test@example.com",
        api_key="testkey",
    )


def test_init_creates_directories(temp_staging: Path) -> None:
    PubMedScraper(staging_dir=temp_staging)
    assert (temp_staging / "xml").exists()
    assert (temp_staging / "metadata").exists()


def test_search_empty_query(scraper: PubMedScraper) -> None:
    assert scraper.search("") == []
    assert scraper.search("   ") == []


@patch("requests.get")
def test_search_success(mock_get: MagicMock, scraper: PubMedScraper) -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "esearchresult": {
            "idlist": ["38123456", "38123457"]
        }
    }
    mock_get.return_value = mock_resp

    pmids = scraper.search("Epstein-Barr Virus", max_results=2)

    assert pmids == ["38123456", "38123457"]
    mock_get.assert_called_once()
    _, kwargs = mock_get.call_args
    assert kwargs["params"]["term"] == "Epstein-Barr Virus"
    assert kwargs["params"]["retmax"] == 2
    assert kwargs["params"]["email"] == "test@example.com"
    assert kwargs["params"]["api_key"] == "testkey"


@patch("requests.get")
def test_fetch_metadata(mock_get: MagicMock, scraper: PubMedScraper) -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "result": {
            "uids": ["38123456"],
            "38123456": {
                "uid": "38123456",
                "title": "EBV B cell interaction",
                "source": "Journal of Virology",
                "pubdate": "2024 May 15",
                "authors": [{"name": "Smith JA"}, {"name": "Doe J"}],
                "articleids": [
                    {"idtype": "doi", "value": "10.1128/jvi.00001-24"},
                    {"idtype": "pmc", "value": "10987654"},
                ],
            },
        }
    }
    mock_get.return_value = mock_resp

    metadata = scraper.fetch_metadata(["38123456"])

    assert "38123456" in metadata
    item = metadata["38123456"]
    assert item["pmid"] == "38123456"
    assert item["pmcid"] == "PMC10987654"
    assert item["doi"] == "10.1128/jvi.00001-24"
    assert item["title"] == "EBV B cell interaction"
    assert item["journal"] == "Journal of Virology"
    assert item["publication_date"] == "2024 May 15"
    assert item["authors"] == ["Smith JA", "Doe J"]


@patch("requests.get")
def test_fetch_pmc_xml_success(mock_get: MagicMock, scraper: PubMedScraper) -> None:
    sample_xml = (
        "<article><front><article-title>Test</article-title></front>"
        "<body><p>Text</p></body></article>"
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = sample_xml
    mock_get.return_value = mock_resp

    xml_content = scraper.fetch_pmc_xml("PMC10987654")
    assert xml_content == sample_xml


@patch("requests.get")
def test_fetch_pmc_xml_failure(mock_get: MagicMock, scraper: PubMedScraper) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "<error>PMC ID not found</error>"
    mock_get.return_value = mock_resp

    assert scraper.fetch_pmc_xml("PMC9999999") is None

    # Test HTTP error response
    mock_resp.status_code = 404
    assert scraper.fetch_pmc_xml("PMC9999999") is None

    # Test request exception
    mock_get.side_effect = requests.RequestException("Connection error")
    assert scraper.fetch_pmc_xml("PMC9999999") is None


@patch.object(PubMedScraper, "search")
@patch.object(PubMedScraper, "fetch_metadata")
@patch.object(PubMedScraper, "fetch_pmc_xml")
@patch.object(PubMedScraper, "fetch_pubmed_abstract")
def test_scrape_workflow(
    mock_abstract: MagicMock,
    mock_xml: MagicMock,
    mock_meta: MagicMock,
    mock_search: MagicMock,
    scraper: PubMedScraper,
) -> None:
    mock_search.return_value = ["38123456", "38123457"]
    mock_meta.return_value = {
        "38123456": {
            "pmid": "38123456",
            "pmcid": "PMC10987654",
            "doi": "10.1128/jvi.00001-24",
            "title": "Article 1",
            "journal": "JVI",
            "publication_date": "2024",
            "authors": ["Author 1"],
        },
        "38123457": {
            "pmid": "38123457",
            "pmcid": None,
            "doi": "10.1128/jvi.00002-24",
            "title": "Article 2",
            "journal": "JVI",
            "publication_date": "2024",
            "authors": ["Author 2"],
        },
    }

    # First article has full XML, second article does not
    mock_xml.side_effect = ["<article>Fulltext 1</article>", None]
    mock_abstract.return_value = "Abstract for article 2"

    result = scraper.scrape("Epstein-Barr Virus", max_results=2)

    assert result["total_found"] == 2
    assert len(result["xml_saved"]) == 1
    assert len(result["metadata_saved"]) == 1

    xml_file = Path(result["xml_saved"][0])
    assert xml_file.name == "38123456.xml"
    assert xml_file.read_text() == "<article>Fulltext 1</article>"

    meta_file = Path(result["metadata_saved"][0])
    assert meta_file.name == "38123457.json"
    saved_meta = json.loads(meta_file.read_text())
    assert saved_meta["pmid"] == "38123457"
    assert saved_meta["abstract"] == "Abstract for article 2"


@patch.object(PubMedScraper, "scrape")
def test_cli_main(
    mock_scrape: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_scrape.return_value = {
        "query": "EBV",
        "total_found": 1,
        "xml_saved": ["/tmp/xml/123.xml"],
        "metadata_saved": [],
    }

    test_args = [
        "pubmed_scraper.py",
        "--query",
        "EBV",
        "--max-results",
        "5",
        "--staging-dir",
        str(tmp_path),
    ]
    monkeypatch.setattr("sys.argv", test_args)

    main()

    mock_scrape.assert_called_once_with("EBV", max_results=5)
