"""Tests for biomedical NER extractor module (Bern2 API and SciSpacy fallback)."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from app.processing.ner_extractor import NERExtractor, NERExtractorError


def test_extract_empty_or_whitespace_text():
    extractor = NERExtractor()
    assert extractor.extract("") == []
    assert extractor.extract("   \n\t ") == []


def test_bern2_url_formatting():
    ext1 = NERExtractor(bern2_url="https://bern2.kbsg.kaist.ac.kr/api")
    assert ext1.bern2_endpoint == "https://bern2.kbsg.kaist.ac.kr/api/plain"

    ext2 = NERExtractor(bern2_url="https://bern2.kbsg.kaist.ac.kr/api/plain")
    assert ext2.bern2_endpoint == "https://bern2.kbsg.kaist.ac.kr/api/plain"


@patch("requests.post")
def test_extract_bern2_success(mock_post, monkeypatch):
    monkeypatch.setenv("ENABLE_BERN2", "true")
    text = "EBNA-1 expression in LCLs causes lymphoma when treated with cisplatin."

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "text": text,
        "annotations": [
            {
                "obj": "gene",
                "prob": 0.997,
                "span": {"begin": 0, "end": 6},
                "id": ["HGNC:3236", "NCBIgene:5126"],
            },
            {
                "obj": "cell_type",
                "prob": 0.884,
                "span": {"begin": 21, "end": 25},
                "id": ["CL:0000236"],
            },
            {
                "obj": "disease",
                "prob": 0.999,
                "span": {"begin": 33, "end": 41},
                "id": ["C0596328"],
            },
            {
                "obj": "chemical",
                "prob": 0.950,
                "span": {"begin": 60, "end": 69},
                "id": ["MESH:D002945"],
            },
            {
                "obj": "species",  # Should be filtered out
                "prob": 0.99,
                "span": {"begin": 0, "end": 4},
                "id": ["NCBI:9606"],
            },
        ],
    }
    mock_post.return_value = mock_response

    extractor = NERExtractor()
    results = extractor.extract(text)

    assert len(results) == 4

    assert results[0] == {
        "text": "EBNA-1",
        "entity_type": "GENE",
        "confidence": 0.997,
        "raw_id": "HGNC:3236",
    }
    assert results[1] == {
        "text": "LCLs",
        "entity_type": "CELL_TYPE",
        "confidence": 0.884,
        "raw_id": "CL:0000236",
    }
    assert results[2] == {
        "text": "lymphoma",
        "entity_type": "DISEASE",
        "confidence": 0.999,
        "raw_id": "C0596328",
    }
    assert results[3] == {
        "text": "cisplatin",
        "entity_type": "CHEMICAL",
        "confidence": 0.950,
        "raw_id": "MESH:D002945",
    }


@patch("requests.post")
@patch("time.sleep")
def test_extract_bern2_retry_success(mock_sleep, mock_post, monkeypatch):
    monkeypatch.setenv("ENABLE_BERN2", "true")
    text = "EBNA-1 causes lymphoma."

    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    mock_resp_429.headers = {}

    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.json.return_value = {
        "text": text,
        "annotations": [
            {
                "obj": "disease",
                "prob": 0.99,
                "span": {"begin": 14, "end": 22},
                "id": ["C0596328"],
            }
        ],
    }

    mock_post.side_effect = [mock_resp_429, mock_resp_200]

    extractor = NERExtractor(max_retries=3, backoff_factor=0.01)
    results = extractor.extract(text)

    assert len(results) == 1
    assert results[0]["entity_type"] == "DISEASE"
    assert mock_post.call_count == 2
    assert mock_sleep.call_count == 1


@patch("requests.post")
def test_extract_fallback_to_scispacy(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError("Bern2 unreachable")

    mock_ent1 = MagicMock()
    mock_ent1.text = "lymphoma"
    mock_ent1.label_ = "DISEASE"

    mock_ent2 = MagicMock()
    mock_ent2.text = "ganciclovir"
    mock_ent2.label_ = "CHEMICAL"

    mock_doc = MagicMock()
    mock_doc.ents = [mock_ent1, mock_ent2]

    mock_nlp = MagicMock()
    mock_nlp.return_value = mock_doc

    extractor = NERExtractor(spacy_model="en_ner_bc5cdr_md")

    with patch.object(extractor, "_get_spacy_nlp", return_value=mock_nlp):
        results = extractor.extract("EBV-associated lymphoma treated with ganciclovir.")

    assert len(results) == 2
    assert results[0] == {
        "text": "lymphoma",
        "entity_type": "DISEASE",
        "confidence": 0.8,
        "raw_id": "",
    }
    assert results[1] == {
        "text": "ganciclovir",
        "entity_type": "CHEMICAL",
        "confidence": 0.8,
        "raw_id": "",
    }


@patch("requests.post")
def test_extract_both_fail_raises_error(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError("Bern2 API offline")

    extractor = NERExtractor(spacy_model="invalid_model_name")

    with patch.object(
        extractor, "_get_spacy_nlp", side_effect=RuntimeError("Model not found")
    ):
        with pytest.raises(NERExtractorError) as exc_info:
            extractor.extract("EBNA-1 expression")

    assert "Both Bern2 API and SciSpacy fallback failed" in str(exc_info.value)
