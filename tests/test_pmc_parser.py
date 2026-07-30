"""Unit tests for PMCXMLParser."""

from pathlib import Path
import pytest
from app.ingestion.pmc_parser import PMCXMLParser


SAMPLE_JATS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<article article-type="research-article">
  <front>
    <journal-meta>
      <journal-title-group>
        <journal-title>Journal of Virology</journal-title>
      </journal-title-group>
    </journal-meta>
    <article-meta>
      <article-id pub-id-type="doi">10.1128/JVI.00001-24</article-id>
      <article-id pub-id-type="pmid">38123456</article-id>
      <article-id pub-id-type="pmc">PMC10987654</article-id>
      <title-group>
        <article-title>Epstein-Barr Virus Latency and Host Interactions</article-title>
      </title-group>
      <contrib-group>
        <contrib contrib-type="author">
          <name>
            <surname>Smith</surname>
            <given-names>John A.</given-names>
          </name>
        </contrib>
        <contrib contrib-type="author">
          <name>
            <surname>Doe</surname>
            <given-names>Jane</given-names>
          </name>
        </contrib>
      </contrib-group>
      <pub-date pub-type="epub">
        <day>15</day>
        <month>05</month>
        <year>2024</year>
      </pub-date>
    </article-meta>
  </front>
  <body>
    <sec sec-type="intro">
      <title>Introduction</title>
      <p>Epstein-Barr virus (EBV) is a human gammaherpesvirus associated with cancers.</p>
      <p>Primary infection often leads to infectious mononucleosis.</p>
    </sec>
    <sec sec-type="methods">
      <title>Materials and Methods</title>
      <p>Cell lines were maintained in RPMI 1640 supplemented with 10% FBS.</p>
      <sec>
        <title>Cell Culture and Infection</title>
        <p>B cells were infected with wild-type EBV strain Akata at MOI of 5.</p>
      </sec>
    </sec>
  </body>
  <back>
    <ref-list>
      <ref id="B1">
        <element-citation publication-type="journal">
          <person-group person-group-type="author">
            <name>
              <surname>Rickinson</surname>
              <given-names>Alan B.</given-names>
            </name>
            <name>
              <surname>Kieff</surname>
              <given-names>Elliott</given-names>
            </name>
          </person-group>
          <article-title>Epstein-Barr Virus and its Replication.</article-title>
          <source>Fields Virology</source>
          <year>2007</year>
          <pub-id pub-id-type="doi">10.1016/B978-0-12-345678-9.00001-0</pub-id>
        </element-citation>
      </ref>
      <ref id="B2">
        <element-citation publication-type="journal">
          <person-group person-group-type="author">
            <name>
              <surname>Young</surname>
              <given-names>L. S.</given-names>
            </name>
          </person-group>
          <article-title>Epstein-Barr virus: 40 years on</article-title>
          <source>Nature Reviews Cancer</source>
          <year>2004</year>
        </element-citation>
      </ref>
    </ref-list>
  </back>
</article>
"""

NAMESPACED_JATS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<article xmlns="http://jats.nlm.nih.gov/ns/archiving/1.2/" xmlns:xlink="http://www.w3.org/1999/xlink">
  <front>
    <journal-meta>
      <journal-title-group>
        <journal-title>Nature Microbiology</journal-title>
      </journal-title-group>
    </journal-meta>
    <article-meta>
      <article-id pub-id-type="doi">10.1038/s41564-023-00000-x</article-id>
      <title-group>
        <article-title>Targeted Analysis of EBV MicroRNAs</article-title>
      </title-group>
      <contrib-group>
        <contrib contrib-type="author">
          <name>
            <surname>Taylor</surname>
            <given-names>Grace</given-names>
          </name>
        </contrib>
      </contrib-group>
      <pub-date pub-type="ppub">
        <year>2023</year>
      </pub-date>
    </article-meta>
  </front>
  <body>
    <sec>
      <title>Results</title>
      <p>MicroRNA expression was quantified by RT-qPCR.</p>
    </sec>
  </body>
</article>
"""


def test_parse_sample_pmc_xml():
    parser = PMCXMLParser()
    result = parser.parse(SAMPLE_JATS_XML)

    # Metadata assertions
    meta = result["metadata"]
    assert meta["title"] == "Epstein-Barr Virus Latency and Host Interactions"
    assert meta["journal"] == "Journal of Virology"
    assert meta["doi"] == "10.1128/JVI.00001-24"
    assert meta["pmid"] == "38123456"
    assert meta["pmcid"] == "PMC10987654"
    assert meta["publication_date"] == "2024-05-15"
    assert meta["authors"] == ["John A. Smith", "Jane Doe"]

    # Text chunks assertions
    chunks = result["text_chunks"]
    assert len(chunks) == 4
    assert chunks[0] == {
        "section": "Introduction",
        "text": "Epstein-Barr virus (EBV) is a human gammaherpesvirus associated with cancers.",
    }
    assert chunks[1] == {
        "section": "Introduction",
        "text": "Primary infection often leads to infectious mononucleosis.",
    }
    assert chunks[2] == {
        "section": "Materials and Methods",
        "text": "Cell lines were maintained in RPMI 1640 supplemented with 10% FBS.",
    }
    assert chunks[3] == {
        "section": "Materials and Methods > Cell Culture and Infection",
        "text": "B cells were infected with wild-type EBV strain Akata at MOI of 5.",
    }

    # References assertions
    refs = result["references"]
    assert len(refs) == 2
    assert refs[0] == {
        "doi": "10.1016/B978-0-12-345678-9.00001-0",
        "title": "Epstein-Barr Virus and its Replication",
        "authors": ["Alan B. Rickinson", "Elliott Kieff"],
        "journal": "Fields Virology",
        "year": "2007",
    }
    assert refs[1] == {
        "doi": None,
        "title": "Epstein-Barr virus: 40 years on",
        "authors": ["L. S. Young"],
        "journal": "Nature Reviews Cancer",
        "year": "2004",
    }


def test_parse_namespaced_xml():
    parser = PMCXMLParser()
    result = parser.parse(NAMESPACED_JATS_XML)

    meta = result["metadata"]
    assert meta["title"] == "Targeted Analysis of EBV MicroRNAs"
    assert meta["journal"] == "Nature Microbiology"
    assert meta["doi"] == "10.1038/s41564-023-00000-x"
    assert meta["publication_date"] == "2023"
    assert meta["authors"] == ["Grace Taylor"]

    chunks = result["text_chunks"]
    assert len(chunks) == 1
    assert chunks[0]["section"] == "Results"
    assert chunks[0]["text"] == "MicroRNA expression was quantified by RT-qPCR."


def test_parse_invalid_xml_raises_value_error():
    parser = PMCXMLParser()
    with pytest.raises(ValueError, match="Invalid XML content provided"):
        parser.parse("<article><front><unclosed-tag></front></article>")


def test_parse_empty_content_raises_value_error():
    parser = PMCXMLParser()
    with pytest.raises(ValueError, match="Empty XML content provided"):
        parser.parse("")


def test_parse_missing_body_or_refs():
    xml_no_body = """<article>
      <front>
        <article-meta>
          <title-group><article-title>Minimal Paper</article-title></title-group>
        </article-meta>
      </front>
    </article>"""
    parser = PMCXMLParser()
    result = parser.parse(xml_no_body)
    assert result["metadata"]["title"] == "Minimal Paper"
    assert result["text_chunks"] == []
    assert result["references"] == []


def test_parse_file_path(tmp_path: Path):
    file_path = tmp_path / "test_article.xml"
    file_path.write_text(SAMPLE_JATS_XML, encoding="utf-8")

    parser = PMCXMLParser()
    result = parser.parse(file_path)

    assert result["metadata"]["title"] == "Epstein-Barr Virus Latency and Host Interactions"
    assert len(result["text_chunks"]) == 4
    assert len(result["references"]) == 2
