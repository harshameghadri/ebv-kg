"""Unit tests for ClaudeSynthesisClient."""

from unittest.mock import MagicMock, patch

import pytest

from app.synthesis.llm import ClaudeSynthesisClient


def test_synthesize_empty_context():
    """Verify empty retrieval contexts return early with 'I do not know'."""
    client = ClaudeSynthesisClient(api_key="test_key")
    res = client.synthesize(
        query="Any question?", retrieved_chunks=[], graph_context=""
    )
    assert res["answer"] == "I do not know"
    assert res["confidence"] == 0.0
    assert res["citations"] == []


@patch("app.synthesis.llm.Anthropic")
def test_synthesize_success_flat_metadata(mock_anthropic_class):
    """Verify successful synthesis, citations, and prompt construction."""
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client
    
    mock_response = MagicMock()
    json_text = (
        '{"answer": "EBV is associated with Burkitt lymphoma [1].", '
        '"confidence": 0.95}'
    )
    mock_response.content = [
        MagicMock(text=json_text)
    ]
    mock_client.messages.create.return_value = mock_response
    
    client = ClaudeSynthesisClient(
        api_key="test_key", model="claude-3-5-sonnet-20240620"
    )
    
    chunks = [
        {
            "id": "chunk-burkitt",
            "content": "EBV association with Burkitt lymphoma is well documented.",
            "pmid": "123456",
            "doi": "10.1000/burkitt"
        }
    ]
    
    res = client.synthesize(
        query="What is EBV associated with?",
        retrieved_chunks=chunks,
        graph_context="EBV -ASSOCIATED_WITH-> Burkitt Lymphoma"
    )
    
    # Assert return structure
    assert res["answer"] == "EBV is associated with Burkitt lymphoma [1]."
    assert res["confidence"] == 0.95
    assert len(res["citations"]) == 1
    
    citation = res["citations"][0]
    assert citation["source_index"] == 1
    assert citation["chunk_id"] == "chunk-burkitt"
    assert citation["pmid"] == "123456"
    assert citation["doi"] == "10.1000/burkitt"
    
    # Verify API invocation details
    mock_client.messages.create.assert_called_once()
    _, kwargs = mock_client.messages.create.call_args
    assert kwargs["model"] == "claude-3-5-sonnet-20240620"
    assert kwargs["max_tokens"] == 4096
    
    user_prompt = kwargs["messages"][0]["content"]
    assert "What is EBV associated with?" in user_prompt
    assert "Chunk ID: chunk-burkitt" in user_prompt
    assert "PMID: 123456 | DOI: 10.1000/burkitt" in user_prompt
    assert "EBV association with Burkitt lymphoma is well documented." in user_prompt
    assert "EBV -ASSOCIATED_WITH-> Burkitt Lymphoma" in user_prompt


@patch("app.synthesis.llm.Anthropic")
def test_synthesize_nested_metadata(mock_anthropic_class):
    """Verify that chunks with nested metadata structures are properly resolved."""
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(text='{"answer": "EBV gp350 binds to CD21 [1].", "confidence": 0.9}')
    ]
    mock_client.messages.create.return_value = mock_response
    
    client = ClaudeSynthesisClient(api_key="test_key")
    
    chunks = [
        {
            "metadata": {
                "id": "chunk-nested-id",
                "pmid": "78910",
                "doi": "10.1000/nested"
            },
            "content": "The EBV glycoprotein gp350 binds CD21 on B cells."
        }
    ]
    
    res = client.synthesize(query="What does gp350 bind?", retrieved_chunks=chunks)
    
    assert res["answer"] == "EBV gp350 binds to CD21 [1]."
    assert res["confidence"] == 0.9
    assert len(res["citations"]) == 1
    
    citation = res["citations"][0]
    assert citation["source_index"] == 1
    assert citation["chunk_id"] == "chunk-nested-id"
    assert citation["pmid"] == "78910"
    assert citation["doi"] == "10.1000/nested"


@patch("app.synthesis.llm.Anthropic")
def test_synthesize_markdown_json_wrapping(mock_anthropic_class):
    """Verify that json blocks wrapped in markdown are parsed correctly."""
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client
    
    mock_response = MagicMock()
    wrapped_json = (
        "```json\n"
        "{\n"
        '  "answer": "EBV infects epithelial cells [1].",\n'
        '  "confidence": "0.80"\n'
        "}\n"
        "```"
    )
    mock_response.content = [
        MagicMock(text=wrapped_json)
    ]
    mock_client.messages.create.return_value = mock_response
    
    client = ClaudeSynthesisClient(api_key="test_key")
    chunks = [
        {
            "id": "chunk-epi",
            "content": "EBV infects epithelial cells",
            "pmid": "111"
        }
    ]
    
    res = client.synthesize(
        query="Does EBV infect epithelial cells?",
        retrieved_chunks=chunks
    )
    assert res["answer"] == "EBV infects epithelial cells [1]."
    assert res["confidence"] == 0.80
    assert len(res["citations"]) == 1
    assert res["citations"][0]["chunk_id"] == "chunk-epi"


@patch("app.synthesis.llm.Anthropic")
def test_synthesize_multiple_citations_mapping(mock_anthropic_class):
    """Verify multiple citations are correctly gathered and resolved."""
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client
    
    mock_response = MagicMock()
    json_text = (
        "{\n"
        '  "answer": "EBV genome is circular [1]. '
        'Latency type III has all genes expressed [2].",\n'
        '  "confidence": 0.99,\n'
        '  "citations": [{"citation_id": 1}, {"citation_id": 2}]\n'
        "}"
    )
    mock_response.content = [
        MagicMock(text=json_text)
    ]
    mock_client.messages.create.return_value = mock_response
    
    client = ClaudeSynthesisClient(api_key="test_key")
    chunks = [
        {"id": "chunk-1", "content": "EBV genome is circular", "pmid": "101"},
        {"id": "chunk-2", "content": "Latency III has all genes", "pmid": "102"},
        {"id": "chunk-3", "content": "Unrelated chunk", "pmid": "103"}
    ]
    
    res = client.synthesize(
        query="Explain EBV circular genome and latency.",
        retrieved_chunks=chunks
    )
    assert len(res["citations"]) == 2
    assert res["citations"][0]["source_index"] == 1
    assert res["citations"][0]["chunk_id"] == "chunk-1"
    assert res["citations"][1]["source_index"] == 2
    assert res["citations"][1]["chunk_id"] == "chunk-2"


@patch("app.synthesis.llm.Anthropic")
def test_synthesize_api_failure(mock_anthropic_class):
    """Verify client handles API failures by raising a RuntimeError."""
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client
    mock_client.messages.create.side_effect = Exception("API connection timed out")
    
    client = ClaudeSynthesisClient(api_key="test_key")
    with pytest.raises(RuntimeError, match="Failed to call Anthropic API"):
        client.synthesize("query", [{"content": "some context"}])


@patch("app.synthesis.llm.Anthropic")
def test_synthesize_parse_failure(mock_anthropic_class):
    """Verify client raises a ValueError if Claude's response is not parseable JSON."""
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(text="Sorry, I could not find any relevant information.")
    ]
    mock_client.messages.create.return_value = mock_response
    
    client = ClaudeSynthesisClient(api_key="test_key")
    with pytest.raises(ValueError, match="Failed to parse Claude response"):
        client.synthesize("query", [{"content": "some context"}])
