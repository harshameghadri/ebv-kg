import os
import re
from typing import Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Ensure HF_TOKEN is mapped to HUGGINGFACE_HUB_TOKEN for huggingface_hub / sentence_transformers
if os.getenv("HF_TOKEN") and not os.getenv("HUGGINGFACE_HUB_TOKEN"):
    os.environ["HUGGINGFACE_HUB_TOKEN"] = os.getenv("HF_TOKEN")

class EmbeddingClient:
    """
    Local embedding client utilizing sentence-transformers or FlagEmbedding.
    Supports lazy loading and automatic device placement (CUDA/MPS/CPU).
    """
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the embedding client.
        
        Args:
            model_name: Optional name of the model to use. Defaults to env var EMBEDDINGS_MODEL
                        or 'allenai/specter2' if not specified.
            device: Optional torch device string. If None, auto-detected.
        """
        self.model_name = model_name or os.getenv("EMBEDDINGS_MODEL", "allenai/specter2")
        self._device = device
        self._model = None
        self._flag_model = None
        self._is_bge_m3 = "bge-m3" in self.model_name.lower()

    @property
    def device(self) -> str:
        """Auto-detect and return the active device."""
        if self._device is None:
            import torch
            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        return self._device

    @property
    def model(self) -> Any:
        """Lazy load the sentence-transformers model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            try:
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as e:
                fallback_name = "BAAI/bge-m3" if "bge" in self.model_name.lower() else "BAAI/bge-large-en-v1.5"
                import logging
                logging.getLogger(__name__).warning(
                    "Failed to load embedding model '%s' (%s). Falling back to '%s'.",
                    self.model_name, e, fallback_name
                )
                self._model = SentenceTransformer(fallback_name, device=self.device)
        return self._model


    def _get_bge_m3_flag_model(self) -> Any:
        """Lazy load the BGEM3FlagModel if FlagEmbedding is available."""
        if self._flag_model is None:
            from FlagEmbedding import BGEM3FlagModel
            use_fp16 = self.device == "cuda"
            self._flag_model = BGEM3FlagModel(self.model_name, use_fp16=use_fp16, device=self.device)
        return self._flag_model

    def embed_query(self, text: str) -> list[float]:
        """
        Generate dense embedding vector for a single query.
        
        Args:
            text: Query text.
            
        Returns:
            A list of floats representing the dense embedding vector.
        """
        if not text:
            return []
        model = self.model
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate dense embedding vectors for a list of document chunks.
        Applies token sorting optimization to minimize padding overhead.
        
        Args:
            texts: List of document chunks.
            
        Returns:
            A list of lists of floats representing the dense embedding vectors.
        """
        if not texts:
            return []
        
        # Token sorting optimization: sort by length before encoding to minimize padding
        indexed_texts = sorted(enumerate(texts), key=lambda x: len(x[1]))
        sorted_indices = [x[0] for x in indexed_texts]
        sorted_texts = [x[1] for x in indexed_texts]
        
        model = self.model
        sorted_embeddings = model.encode(sorted_texts, convert_to_numpy=True)
        
        # Restore original order
        embeddings = [None] * len(texts)
        for original_idx, emb in zip(sorted_indices, sorted_embeddings):
            embeddings[original_idx] = emb.tolist()
            
        return embeddings

    def embed_query_sparse(self, text: str) -> dict[str, float]:
        """
        Generate sparse lexical weights for a single text.
        
        Args:
            text: Text to embed.
            
        Returns:
            A dictionary mapping tokens to their float weights.
        """
        if not text:
            return {}
            
        if self._is_bge_m3:
            try:
                model = self._get_bge_m3_flag_model()
                output = model.encode([text], return_sparse=True)
                weights = output['lexical_weights'][0]
                
                decoded_weights = {}
                for k, v in weights.items():
                    if isinstance(k, (int, str)):
                        token = model.tokenizer.decode([int(k)]) if isinstance(k, int) or k.isdigit() else str(k)
                        token = token.replace('Ġ', '').replace(' ', '').strip()
                        if token:
                            decoded_weights[token] = float(v)
                return decoded_weights
            except Exception:
                return self._compute_sparse_fallback(text)
        else:
            return self._compute_sparse_fallback(text)

    def embed_documents_sparse(self, texts: list[str]) -> list[dict[str, float]]:
        """
        Generate sparse lexical weights for a list of document chunks.
        Applies token sorting optimization to minimize padding overhead.
        
        Args:
            texts: List of document chunks.
            
        Returns:
            A list of dictionaries mapping tokens to their float weights.
        """
        if not texts:
            return []
            
        if self._is_bge_m3:
            try:
                # Token sorting optimization
                indexed_texts = sorted(enumerate(texts), key=lambda x: len(x[1]))
                sorted_indices = [x[0] for x in indexed_texts]
                sorted_texts = [x[1] for x in indexed_texts]
                
                model = self._get_bge_m3_flag_model()
                output = model.encode(sorted_texts, return_sparse=True)
                lexical_weights = output['lexical_weights']
                
                sorted_results = []
                for weights in lexical_weights:
                    decoded_weights = {}
                    for k, v in weights.items():
                        if isinstance(k, (int, str)):
                            token = model.tokenizer.decode([int(k)]) if isinstance(k, int) or k.isdigit() else str(k)
                            token = token.replace('Ġ', '').replace(' ', '').strip()
                            if token:
                                decoded_weights[token] = float(v)
                    sorted_results.append(decoded_weights)
                
                # Restore original order
                results = [None] * len(texts)
                for original_idx, res in zip(sorted_indices, sorted_results):
                    results[original_idx] = res
                return results
            except Exception:
                return [self._compute_sparse_fallback(text) for text in texts]
        else:
            return [self._compute_sparse_fallback(text) for text in texts]

    def _compute_sparse_fallback(self, text: str) -> dict[str, float]:
        """
        Fallback method to compute token frequencies as sparse weights
        when BGE-M3 sparse weights cannot be generated.
        """
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return {}
        # Simple term frequency
        return {word: float(words.count(word)) for word in set(words)}
