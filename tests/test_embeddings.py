import os
import sys
from unittest.mock import MagicMock, patch

# Inject mock modules into sys.modules to prevent real torch/sentence_transformers
# imports during testing, avoiding any local segmentation faults or GPU loading overhead.
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
mock_torch.backends.mps.is_available.return_value = False

mock_st = MagicMock()
mock_st_class = MagicMock()
mock_st.SentenceTransformer = mock_st_class

mock_fe = MagicMock()
mock_fe_model = MagicMock()
mock_fe.BGEM3FlagModel = mock_fe_model

sys.modules['torch'] = mock_torch
sys.modules['sentence_transformers'] = mock_st
sys.modules['FlagEmbedding'] = mock_fe

# Now import the client, which executes safely without importing real dependencies
from app.retrieval.embeddings import EmbeddingClient

import pytest
import numpy as np

@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset the mock objects before each test."""
    mock_torch.reset_mock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False
    mock_st_class.reset_mock()
    mock_fe_model.reset_mock()

def test_client_init():
    """Verify that EmbeddingClient initializes with correct defaults and environment variables."""
    # Test default initialization
    client = EmbeddingClient()
    assert client.model_name == "all-MiniLM-L6-v2"
    assert client._device is None
    assert client._model is None
    assert client._flag_model is None
    assert not client._is_bge_m3

    # Test initialization with customized model name
    client_custom = EmbeddingClient(model_name="allenai/specter2")
    assert client_custom.model_name == "allenai/specter2"
    assert not client_custom._is_bge_m3

    # Test initialization with BGE-M3 model name detection
    client_bge = EmbeddingClient(model_name="BAAI/bge-m3")
    assert client_bge.model_name == "BAAI/bge-m3"
    assert client_bge._is_bge_m3

    # Test respect for EMBEDDINGS_MODEL environment variable
    with patch.dict(os.environ, {"EMBEDDINGS_MODEL": "custom-env-model"}):
        client_env = EmbeddingClient()
        assert client_env.model_name == "custom-env-model"

def test_device_detection():
    """Verify device placement logic handles cuda, mps, and cpu."""
    # Test default CPU fallback
    client = EmbeddingClient()
    assert client.device == "cpu"

    # Test CUDA detection
    mock_torch.cuda.is_available.return_value = True
    client_cuda = EmbeddingClient()
    assert client_cuda.device == "cuda"

    # Test MPS detection
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = True
    client_mps = EmbeddingClient()
    assert client_mps.device == "mps"

    # Test explicit device override
    client_override = EmbeddingClient(device="custom-device")
    assert client_override.device == "custom-device"

def test_embed_query():
    """Verify single query embedding returns correct list format from mocked model."""
    mock_model_instance = MagicMock()
    mock_model_instance.encode.return_value = np.array([0.1, 0.2, 0.3])
    mock_st_class.return_value = mock_model_instance

    client = EmbeddingClient()
    embedding = client.embed_query("test query")

    assert isinstance(embedding, list)
    assert embedding == [0.1, 0.2, 0.3]
    mock_model_instance.encode.assert_called_once_with("test query", convert_to_numpy=True)

def test_embed_documents():
    """Verify document chunk embedding works and token sorting logic preserves ordering."""
    mock_model_instance = MagicMock()
    
    # In token sorting, inputs will be sorted by length:
    # "a" (length 1)
    # "short" (length 5)
    # "very long text" (length 14)
    # Mock encode to return embeddings corresponding to these sorted inputs
    mock_model_instance.encode.return_value = np.array([
        [1.0, 1.0],  # for "a"
        [2.0, 2.0],  # for "short"
        [3.0, 3.0]   # for "very long text"
    ])
    mock_st_class.return_value = mock_model_instance

    client = EmbeddingClient()
    documents = ["short", "very long text", "a"]
    embeddings = client.embed_documents(documents)

    assert isinstance(embeddings, list)
    assert len(embeddings) == 3
    # Check that sorting was applied (encode called with sorted inputs)
    mock_model_instance.encode.assert_called_once_with(["a", "short", "very long text"], convert_to_numpy=True)
    
    # Check that original order was correctly restored
    # "short" is at index 0 -> should match [2.0, 2.0]
    # "very long text" is at index 1 -> should match [3.0, 3.0]
    # "a" is at index 2 -> should match [1.0, 1.0]
    assert embeddings[0] == [2.0, 2.0]
    assert embeddings[1] == [3.0, 3.0]
    assert embeddings[2] == [1.0, 1.0]

def test_embed_query_sparse_fallback_non_bge():
    """Verify sparse embedding fallback uses lexical term frequencies for non-BGE-M3 models."""
    client = EmbeddingClient(model_name="all-MiniLM-L6-v2")
    sparse_weights = client.embed_query_sparse("The cells and cells differentiation")

    assert isinstance(sparse_weights, dict)
    assert sparse_weights["cells"] == 2.0
    assert sparse_weights["differentiation"] == 1.0
    assert sparse_weights["and"] == 1.0
    assert sparse_weights["the"] == 1.0

def test_embed_documents_sparse_fallback_non_bge():
    """Verify sparse document embedding fallback works for non-BGE-M3 models."""
    client = EmbeddingClient(model_name="all-MiniLM-L6-v2")
    docs = ["normal cell", "abnormal cell normal"]
    results = client.embed_documents_sparse(docs)

    assert len(results) == 2
    assert results[0] == {"normal": 1.0, "cell": 1.0}
    assert results[1] == {"abnormal": 1.0, "cell": 1.0, "normal": 1.0}

def test_embed_query_sparse_bge_m3():
    """Verify sparse weights extraction decodes token IDs to tokens using BGE-M3 model."""
    mock_bge_instance = MagicMock()
    mock_bge_instance.encode.return_value = {
        "lexical_weights": [{"101": 0.85, "102": 0.45}]
    }
    
    # Mock tokenizer decode behavior
    def mock_decode(token_ids):
        if token_ids == [101]:
            return "Ġcxcr3"
        if token_ids == [102]:
            return "Ġcell"
        return ""
    mock_bge_instance.tokenizer.decode.side_effect = mock_decode
    mock_fe_model.return_value = mock_bge_instance

    client = EmbeddingClient(model_name="BAAI/bge-m3")
    sparse_weights = client.embed_query_sparse("cxcr3 cell")

    assert isinstance(sparse_weights, dict)
    # The 'Ġ' symbol should be stripped
    assert sparse_weights["cxcr3"] == 0.85
    assert sparse_weights["cell"] == 0.45
    mock_bge_instance.encode.assert_called_once_with(["cxcr3 cell"], return_sparse=True)

def test_embed_documents_sparse_bge_m3():
    """Verify BGE-M3 sparse encoding for documents supports token sorting and order restoration."""
    mock_bge_instance = MagicMock()
    
    # inputs sorted: "a", "short", "very long"
    mock_bge_instance.encode.return_value = {
        "lexical_weights": [
            {"1": 0.1}, # for "a"
            {"2": 0.2}, # for "short"
            {"3": 0.3}  # for "very long"
        ]
    }
    
    def mock_decode(token_ids):
        return f"tok_{token_ids[0]}"
    mock_bge_instance.tokenizer.decode.side_effect = mock_decode
    mock_fe_model.return_value = mock_bge_instance

    client = EmbeddingClient(model_name="BAAI/bge-m3")
    docs = ["short", "very long", "a"]
    results = client.embed_documents_sparse(docs)

    assert len(results) == 3
    # Check that sorting was applied
    mock_bge_instance.encode.assert_called_once_with(["a", "short", "very long"], return_sparse=True)
    # Check order restoration
    # index 0: "short" -> tok_2
    # index 1: "very long" -> tok_3
    # index 2: "a" -> tok_1
    assert results[0] == {"tok_2": 0.2}
    assert results[1] == {"tok_3": 0.3}
    assert results[2] == {"tok_1": 0.1}

def test_sparse_fallback_on_error():
    """Verify client falls back to lexical frequency if FlagEmbedding throws an error."""
    mock_bge_instance = MagicMock()
    mock_bge_instance.encode.side_effect = Exception("Inference error")
    mock_fe_model.return_value = mock_bge_instance

    client = EmbeddingClient(model_name="BAAI/bge-m3")
    sparse_weights = client.embed_query_sparse("fallback text")

    # Should fallback gracefully
    assert isinstance(sparse_weights, dict)
    assert sparse_weights["fallback"] == 1.0
    assert sparse_weights["text"] == 1.0
