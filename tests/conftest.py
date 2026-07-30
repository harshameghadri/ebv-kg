"""pytest configuration and global mocks for torch and sentence-transformers.

Injects mock modules into sys.modules before any tests are collected or run,
preventing CPU/GPU memory leaks or segmentation faults in sandbox environments.
"""

import sys
from unittest.mock import MagicMock

# 1. Define global mock objects
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
mock_torch.backends.mps.is_available.return_value = False

mock_st_class = MagicMock()
mock_st = MagicMock()
mock_st.SentenceTransformer = mock_st_class
mock_st.CrossEncoder = MagicMock()

mock_fe_model = MagicMock()
mock_fe = MagicMock()
mock_fe.BGEM3FlagModel = mock_fe_model

# 2. Inject mocks into sys.modules
sys.modules["torch"] = mock_torch
sys.modules["sentence_transformers"] = mock_st
sys.modules["FlagEmbedding"] = mock_fe
