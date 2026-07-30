"""Unit tests for PDFExtractor (Grobid client with PyMuPDF fallback)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import pytest
import requests

from app.ingestion.pdf_extractor import PDFExtractor

SAMPLE_TEI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title level="a" type="main">Epstein-Barr Virus MicroRNAs and Pathogenesis</title>
      </titleStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <title level="a" type="main">Epstein-Barr Virus MicroRNAs and Pathogenesis</title>
            <author>
              <persName>
                <forename type="first">Alice</forename>
                <surname>Smith</surname>
              </persName>
            </author>
            <author>
              <persName>
                <forename type="first">Bob</forename>
                <surname>Jones</surname>
              </persName>
            </author>
            <idno type="DOI">10.1038/s41564-024-00123-x</idno>
            <idno type="PMID">38999999</idno>
          </analytic>
          <monogr>
            <title level="j" type="main">Nature Microbiology</title>
            <imprint>
              <date type="published" when="2024-03-15">2024</date>
            </imprint>
          </monogr>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
    <profileDesc>
      <abstract>
        <p>Epstein-Barr virus encodes multiple non-coding RNAs during latent infection.</p>
      </abstract>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div n="1">
        <head>Introduction</head>
        <p>EBV is associated with Burkitt lymphoma and nasopharyngeal carcinoma.</p>
        <p>MicroRNAs modulate viral and cellular gene expression.</p>
      </div>
      <div n="2">
        <head>Results</head>
        <p>Deep sequencing identified novel viral transcript isoforms.</p>
      </div>
    </body>
    <back>
      <div type="references">
        <listBibl>
          <biblStruct xml:id="b0">
            <analytic>
              <title level="a" type="main">EBV microRNA function in latency</title>
              <author>
                <persName>
                  <forename type="first">Carol</forename>
                  <surname>White</surname>
                </persName>
              </author>
              <idno type="DOI">10.1016/j.cell.2020.01.005</idno>
            </analytic>
            <monogr>
              <title level="j">Cell</title>
              <imprint>
                <date type="published" when="2020">2020</date>
              </imprint>
            </monogr>
          </biblStruct>
        </listBibl>
      </div>
    </back>
  </text>
</TEI>
"""


def create_sample_pdf_bytes() -> bytes:
    """Helper to create a minimal PDF in memory using PyMuPDF."""
    doc = fitz.open()

    # Set basic metadata
    doc.set_metadata({
        "title": "Synthetic EBV Research Paper",
        "author": "David Miller, Eva Green",
    })

    page = doc.new_page()

    # Title line (large font size 18)
    page.insert_text(
        (50, 50),
        "Synthetic EBV Research Paper",
        fontsize=18,
        fontname="helv",
    )
    # DOI and PMID (font size 10)
    page.insert_text(
        (50, 75),
        "DOI: 10.1016/j.jvi.2024.100200 PMID: 37654321 Year: 2024",
        fontsize=10,
    )
    # Header: Introduction (large font size 14)
    page.insert_text((50, 110), "Introduction", fontsize=14)
    # Body text (font size 10)
    page.insert_text(
        (50, 130),
        "Epstein-Barr virus infects over 90% of the adult human population worldwide.",
        fontsize=10,
    )
    page.insert_text(
        (50, 150),
        "The virus establishes lifelong latent infection in B lymphocytes.",
        fontsize=10,
    )

    # Header: Results (large font size 14)
    page.insert_text((50, 190), "Results", fontsize=14)
    page.insert_text(
        (50, 210),
        "Viral latency genes were expressed at high levels in EBV positive cell lines.",
        fontsize=10,
    )

    # Header: References (large font size 14)
    page.insert_text((50, 250), "References", fontsize=14)
    page.insert_text(
        (50, 270),
        "1. Rickinson AB, Kieff E. Epstein-Barr Virus. Fields Virology 2007. DOI: 10.1016/b978-012",
        fontsize=10,
    )

    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def test_grobid_extraction_success():
    extractor = PDFExtractor(grobid_url="http://localhost:8070")
    pdf_bytes = create_sample_pdf_bytes()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = SAMPLE_TEI_XML.encode("utf-8")

    with patch("requests.post", return_value=mock_response) as mock_post:
        result = extractor.parse(pdf_bytes)

        # Assert requests.post call arguments
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://localhost:8070/api/processFulltextDocument"

        # Assert Metadata
        meta = result["metadata"]
        assert meta["title"] == "Epstein-Barr Virus MicroRNAs and Pathogenesis"
        assert meta["authors"] == ["Alice Smith", "Bob Jones"]
        assert meta["doi"] == "10.1038/s41564-024-00123-x"
        assert meta["pmid"] == "38999999"
        assert meta["journal"] == "Nature Microbiology"
        assert meta["publication_date"] == "2024-03-15"
        assert meta["year"] == "2024"

        # Assert Chunks
        chunks = result["chunks"]
        assert len(chunks) == 4
        assert chunks[0] == {
            "section": "Abstract",
            "text": "Epstein-Barr virus encodes multiple non-coding RNAs during latent infection.",
        }
        assert chunks[1] == {
            "section": "Introduction",
            "text": "EBV is associated with Burkitt lymphoma and nasopharyngeal carcinoma.",
        }
        assert chunks[2] == {
            "section": "Introduction",
            "text": "MicroRNAs modulate viral and cellular gene expression.",
        }
        assert chunks[3] == {
            "section": "Results",
            "text": "Deep sequencing identified novel viral transcript isoforms.",
        }

        # Assert alias text_chunks
        assert result["text_chunks"] == chunks

        # Assert References
        refs = result["references"]
        assert len(refs) == 1
        assert refs[0] == {
            "title": "EBV microRNA function in latency",
            "authors": ["Carol White"],
            "journal": "Cell",
            "year": "2020",
            "doi": "10.1016/j.cell.2020.01.005",
        }


def test_grobid_fallback_to_pymupdf_on_connection_error():
    extractor = PDFExtractor(grobid_url="http://localhost:8070")
    pdf_bytes = create_sample_pdf_bytes()

    with patch(
        "requests.post", side_effect=requests.exceptions.ConnectionError("Grobid offline")
    ):
        result = extractor.parse(pdf_bytes)

        # Assert metadata extracted via PyMuPDF fallback
        meta = result["metadata"]
        assert meta["title"] == "Synthetic EBV Research Paper"
        assert meta["authors"] == ["David Miller", "Eva Green"]
        assert meta["doi"] == "10.1016/j.jvi.2024.100200"
        assert meta["pmid"] == "37654321"
        assert meta["year"] == "2024"

        # Assert chunks extracted via PyMuPDF heuristic font grouping
        chunks = result["chunks"]
        assert len(chunks) > 0
        sections = [c["section"] for c in chunks]
        assert "Introduction" in sections or "Results" in sections

        # Assert references
        refs = result["references"]
        assert len(refs) > 0


def test_grobid_fallback_to_pymupdf_on_500_status():
    extractor = PDFExtractor(grobid_url="http://localhost:8070")
    pdf_bytes = create_sample_pdf_bytes()

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    with patch("requests.post", return_value=mock_response):
        result = extractor.parse(pdf_bytes)
        assert result["metadata"]["title"] == "Synthetic EBV Research Paper"
        assert len(result["chunks"]) > 0


def test_custom_grobid_url():
    custom_url = "http://grobid-service:8080/"
    extractor = PDFExtractor(grobid_url=custom_url)
    pdf_bytes = create_sample_pdf_bytes()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = SAMPLE_TEI_XML.encode("utf-8")

    with patch("requests.post", return_value=mock_response) as mock_post:
        extractor.parse(pdf_bytes)
        args, _ = mock_post.call_args
        assert args[0] == "http://grobid-service:8080/api/processFulltextDocument"


def test_file_input_not_found():
    extractor = PDFExtractor()
    with pytest.raises(FileNotFoundError):
        extractor.parse(Path("/non/existent/path/paper.pdf"))


def test_file_path_input(tmp_path: Path):
    file_path = tmp_path / "paper.pdf"
    file_path.write_bytes(create_sample_pdf_bytes())

    extractor = PDFExtractor()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = SAMPLE_TEI_XML.encode("utf-8")

    with patch("requests.post", return_value=mock_response):
        result = extractor.parse(file_path)
        assert result["metadata"]["title"] == "Epstein-Barr Virus MicroRNAs and Pathogenesis"
        assert len(result["chunks"]) == 4
