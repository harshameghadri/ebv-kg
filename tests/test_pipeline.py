"""Unit tests for ETLPipeline."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest
import psycopg

from app.ingestion.pipeline import ETLPipeline


@pytest.fixture
def pipeline() -> ETLPipeline:
    return ETLPipeline(
        pg_dsn="postgresql://user:pass@localhost:5432/dbname",
        lancedb_uri="data/test_lancedb",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        neo4j_database="neo4j",
        staging_dir="data/test_staging",
    )


@patch("app.ingestion.pipeline.psycopg.connect")
@patch("app.ingestion.pipeline.PubMedScraper")
@patch("app.ingestion.pipeline.PMCXMLParser")
@patch("app.ingestion.pipeline.PDFExtractor")
@patch("app.ingestion.pipeline.NERExtractor")
@patch("app.ingestion.pipeline.EntityMapper")
@patch("app.ingestion.pipeline.EmbeddingsPipeline")
@patch("app.ingestion.pipeline.Materializer")
@patch("app.ingestion.pipeline.Neo4jClient")
@patch("app.ingestion.pipeline.LanceDBClient")
def test_run_etl_pipeline_success(
    mock_lancedb_client_cls,
    mock_neo4j_client_cls,
    mock_materializer_cls,
    mock_embeddings_pipeline_cls,
    mock_entity_mapper_cls,
    mock_ner_extractor_cls,
    mock_pdf_extractor_cls,
    mock_pmc_parser_cls,
    mock_pubmed_scraper_cls,
    mock_psycopg_connect,
    pipeline,
    tmp_path,
) -> None:
    # 1. Setup mock connections and cursors
    mock_conn = MagicMock()
    mock_psycopg_connect.return_value = mock_conn

    # 2. Setup PubMedScraper mock
    mock_scraper = MagicMock()
    mock_pubmed_scraper_cls.return_value = mock_scraper
    
    xml_file = tmp_path / "38123456.xml"
    xml_file.write_text("<article>Test XML</article>", encoding="utf-8")
    
    json_file = tmp_path / "38123457.json"
    meta_content = {
        "pmid": "38123457",
        "title": "Test JSON",
        "abstract": "Abstract text",
        "publication_date": "2024"
    }
    json_file.write_text(json.dumps(meta_content), encoding="utf-8")
    
    mock_scraper.scrape.return_value = {
        "total_found": 2,
        "xml_saved": [str(xml_file)],
        "metadata_saved": [str(json_file)]
    }

    # Set staging_dir to tmp_path to use our files
    pipeline.staging_dir = tmp_path

    # 3. Setup PMCXMLParser mock
    mock_pmc_parser = MagicMock()
    mock_pmc_parser_cls.return_value = mock_pmc_parser
    mock_pmc_parser.parse.return_value = {
        "metadata": {"title": "XML Doc", "doi": "10.1000/xml", "pmid": "38123456"},
        "text_chunks": [{"section": "Introduction", "text": "EBV infects cells."}],
        "references": []
    }

    # 4. Setup NERExtractor mock
    mock_ner = MagicMock()
    mock_ner_extractor_cls.return_value = mock_ner
    mock_ner.extract.side_effect = [
        # For XML chunk
        [{"text": "EBV", "entity_type": "GENE", "confidence": 0.9, "raw_id": ""}],
        # For JSON chunk
        [{"text": "Abstract", "entity_type": "CHEMICAL", "confidence": 0.8, "raw_id": ""}]
    ]

    # 5. Setup EntityMapper mock
    mock_mapper = MagicMock()
    mock_entity_mapper_cls.return_value = mock_mapper

    # 6. Setup EmbeddingsPipeline mock
    mock_embeddings = MagicMock()
    mock_embeddings_pipeline_cls.return_value = mock_embeddings
    mock_embeddings.index_pending_chunks.return_value = 10

    # 7. Setup Materializer mock
    mock_materializer = MagicMock()
    mock_materializer_cls.return_value = mock_materializer
    mock_materializer.materialize_graph.return_value = {"entities": 5, "relationships": 3}

    # Run pipeline
    result = pipeline.run_etl_pipeline(query="EBV", max_articles=2)

    # Assertions
    assert result["query"] == "EBV"
    assert result["scraped_count"] == 2
    assert result["processed_docs"] == 2
    assert result["indexed_chunks"] == 10
    assert result["materialized_stats"] == {"entities": 5, "relationships": 3}

    # Verify connection calls
    assert mock_psycopg_connect.call_count == 3  # step 2, step 3, step 4
    
    # Verify Scraper call
    mock_pubmed_scraper_cls.assert_called_once_with(staging_dir=tmp_path)
    mock_scraper.scrape.assert_called_once_with("EBV", max_results=2)

    # Verify XML Parser call
    mock_pmc_parser.parse.assert_called_once_with(xml_file)

    # Verify NER call
    assert mock_ner.extract.call_count == 2
    mock_ner.extract.assert_has_calls([
        call("EBV infects cells."),
        call("Abstract text")
    ])

    # Verify Entity Mapper call
    assert mock_mapper.map_document_content.call_count == 2

    # Verify Embeddings Pipeline call
    mock_embeddings.index_pending_chunks.assert_called_once()

    # Verify Materializer call
    mock_materializer.materialize_graph.assert_called_once()
    called_kwargs = mock_materializer.materialize_graph.call_args.kwargs
    assert called_kwargs["pg_conn"] == mock_conn
    assert called_kwargs["curation_statuses"] == ["APPROVED", "PENDING"]
    assert "doc_ids" in called_kwargs


@patch("app.ingestion.pipeline.psycopg.connect")
@patch("app.ingestion.pipeline.PubMedScraper")
@patch("app.ingestion.pipeline.PMCXMLParser")
@patch("app.ingestion.pipeline.NERExtractor")
def test_run_etl_pipeline_parse_error(
    mock_ner_extractor_cls,
    mock_pmc_parser_cls,
    mock_pubmed_scraper_cls,
    mock_psycopg_connect,
    pipeline,
    tmp_path,
) -> None:
    mock_conn = MagicMock()
    mock_psycopg_connect.return_value = mock_conn

    mock_scraper = MagicMock()
    mock_pubmed_scraper_cls.return_value = mock_scraper
    
    xml_file = tmp_path / "38123456.xml"
    xml_file.write_text("<article>Test XML</article>", encoding="utf-8")
    
    mock_scraper.scrape.return_value = {
        "total_found": 1,
        "xml_saved": [str(xml_file)],
        "metadata_saved": []
    }

    pipeline.staging_dir = tmp_path

    # Make PMCXMLParser fail
    mock_pmc_parser = MagicMock()
    mock_pmc_parser_cls.return_value = mock_pmc_parser
    mock_pmc_parser.parse.side_effect = ValueError("XML Parse Error")

    # Expect pipeline to fail loud and raise the parsing error
    with pytest.raises(ValueError, match="XML Parse Error"):
        pipeline.run_etl_pipeline(query="EBV", max_articles=1)

    # Verify connection was closed
    mock_conn.close.assert_called()
