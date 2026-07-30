# Technical Challenges & Solutions

This document logs significant technical challenges encountered during the development of the EBV Knowledge System and their resolutions.

---

### Challenge 1: JATS XML Parsing Truth-Testing Gotcha

*   **Location**: `app/ingestion/pmc_parser.py`
*   **Problem**: In Python's standard `xml.etree.ElementTree` (and `lxml`), truth-testing an XML element (e.g. `if element:`) returns `False` if that element has no child nodes, even if it contains text (like `<source>Fields Virology</source>`). Chaining multiple element checks using `or` resulted in bibliographic references failing to parse and returning `None`.
*   **Solution**: Replaced implicit truth-testing with explicit `is not None` checks.
    *   *Incorrect Code*:
        ```python
        title = r.find("article-title") or r.find("chapter-title")
        ```
    *   *Correct Code*:
        ```python
        title_el = r.find("article-title")
        if title_el is None:
            title_el = r.find("chapter-title")
        title = title_el.text if title_el is not None else None
        ```
*   **Git Commit**: `a9fb0db`

---

### Challenge 2: Neo4j Mock Assertions Verification Failures

*   **Location**: `tests/test_neo4j_client.py`
*   **Problem**: When mocking database results in Python unit tests, iterating over a mocked result (e.g., `list(result)`) registers internal magic method calls (`__iter__` and `__len__`) on the parent mock object. Asserting execution patterns using `session.run.assert_has_calls(...)` failed because of these extra registered calls.
*   **Solution**: Switched verification to assert directly against the exact SQL/Cypher calls logged in `session.run.call_args_list`.
    *   *Incorrect Code*:
        ```python
        session.run.assert_has_calls([mock.call(expected_query, ...)])
        ```
    *   *Correct Code*:
        ```python
        assert session.run.call_args_list == [mock.call(expected_query, ...)]
        ```
*   **Git Commit**: `a9fb0db`

---

### Challenge 3: LanceDB Directory Renaming Error on Mounted Volumes

*   **Location**: `app/retrieval/vector.py` and `tests/test_vector.py`
*   **Problem**: LanceDB relies on transactional/atomic folder renaming operations. When database files were stored directly inside the `/Volumes/Projects/` mount, LanceDB table creation crashed with:
    ```
    RuntimeError: lance error: LanceError(IO): Generic LocalFileSystem error: Unable to rename file: Operation not supported (os error 45)
    ```
    This occurred because network mount-points do not support local POSIX atomic folder renaming.
*   **Solution**: Modified the runtime database configuration and testing suites to store the database in local filesystem directories (such as `/tmp/`, local system temporary folders using pytest's `tmp_path`, or `~/.gemini/` directories) which are hosted on APFS volumes.
*   **Git Commit**: `a9fb0db`

---

### Challenge 4: LanceDB Dot Metric Similarity Mapping

*   **Location**: `app/retrieval/vector.py`
*   **Problem**: LanceDB returns distances for both `"cosine"` and `"dot"` search metrics as `1.0 - similarity_score`. The initial implementation mapped the dot metric's similarity directly to the distance value (`similarity = distance`), which resulted in exact matches receiving a similarity score of `0.0` instead of `1.0`.
*   **Solution**: Corrected the score translation block so both cosine and dot product metrics are mapped correctly.
    *   *Incorrect Code*:
        ```python
        if metric == "cosine":
            similarity = 1.0 - distance
        elif metric == "dot":
            similarity = distance
        ```
    *   *Correct Code*:
        ```python
        if metric in ("cosine", "dot"):
            similarity = 1.0 - distance
        ```
*   **Git Commit**: `a9fb0db`

---

### Challenge 5: Pytest Mock Injection Collision and Test Isolation Leakage

*   **Location**: `tests/test_embeddings.py` and `tests/test_hybrid.py`
*   **Problem**: Both test modules injected custom mock objects into `sys.modules['torch']` and `sys.modules['sentence_transformers']` globally at import time to prevent downloading heavy model weights or device placement hangs. When pytest collected and ran the tests, the imports in `test_hybrid.py` ran after `test_embeddings.py`'s import-time setup, overwriting the global mocks in `sys.modules` with new instances. As a result, the methods in `EmbeddingClient` looked up the overwritten mocks and failed with assertions (e.g. CUDA device detection and mock method assertions failed due to mock leakage).
*   **Solution**: Factored out global import-time monkeypatching into a shared `tests/conftest.py` file. This configures the `sys.modules` mocks exactly once globally before test collection. Modified both test files to import the shared mock objects (e.g. `mock_torch`, `mock_st`, `mock_fe_model`) from `conftest` rather than re-injecting them.
*   **Git Commit**: `a9fb0db` (and subsequently `fc91d6b`)

