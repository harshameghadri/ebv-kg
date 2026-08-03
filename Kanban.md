# Project Kanban

This document serves as the agile task tracker for the EBV Knowledge System. It contains atomic, goal-oriented tasks with explicit endpoint-specific definitions.

## Backlog

### EPIC01: Ingestion Layer
- [x] **T105**: Build a CLI command to ingest and parse single-cell AnnData (`.h5ad`) files and export marker gene CSVs, cell type labels into PostgreSQL staging (`app/ingestion/anndata_cli.py`).

### EPIC02: Entity Normalization & NER Pipeline
- [x] **T203**: Implement entity mapping module `app/entity_mapper.py` that normalizes raw NER entities to canonical ontology IDs with combined confidence scores.

### EPIC03: Knowledge Graph Materialization
- [x] **T302**: Implement `AgeEngine` using PostgreSQL AGE extension to execute graph operations inside PostgreSQL (`app/materialization/age_engine.py`).
- [x] **T303**: Implement `KuzuEngine` in `app/materialization/kuzu_engine.py` to execute graph operations on the embedded C++ Kùzu engine with fallback support and 2-hop neighborhood path retrieval.
- [x] **T304**: Built graph engine benchmarking script (`app/materialization/benchmark_graph_engines.py`) measuring write throughput and multi-hop Cypher read latency across Neo4jClient, KuzuEngine, and AgeEngine, returning structured JSON metrics.
- [x] **T305**: Implement the materialization pipeline module that reads normalized entities and relationships from PostgreSQL and writes them into the selected graph engine.

### EPIC04: Retrieval & Hybrid Search Layer
- [x] **T401**: Set up LanceDB vector store client in `app/retrieval/vector.py` supporting Arrow-native dense and metadata indexing.
- [x] **T402**: Implement local BGE-M3 embedding client in `app/embeddings.py` to generate dense vector embeddings and sparse lexical weights.
- [x] **T403**: Build hybrid retriever in `app/retrieval/hybrid.py` that queries LanceDB for dense candidates, combines with sparse lexical candidates, and performs cross-encoder reranking using `bge-reranker-v2`.
- [x] **T404**: Implement graph-augmented context retrieval by querying multi-hop entity associations from the materialization graph engine to expand the document search candidates.

### EPIC05: LLM Synthesis & RAG Query Layer
- [x] **T501**: Implemented LLM synthesis client utilizing Claude API to generate factual answers with confidence scoring and citation mapping, verified with comprehensive tests.
- [x] **T502**: Implement FastAPI server with routes for search queries, hybrid semantic retrieval, RAG synthesis, and graph node visualization.
- [x] **T503**: Build evaluation test suite comparing specter2 vs BGE-M3 embedding models on 50+ gold-standard EBV biological queries, calculating recall@K.

### EPIC06: Human Curation & Discovery Loop
- [x] **T601**: Implement LightRAG indexing runner in `discovery/lightrag_runner.py` that runs automated clustering and community detection on the PostgreSQL text corpus.
- [x] **T602**: Build the harvesting script in `discovery/harvest.py` that ranks LightRAG discovery candidates against the canonical KG and promotes the top 20 candidates weekly to the review queue.
- [x] **T603**: Create web-based Curation Dashboard using HTML/CSS/JS (FastAPI frontend) displaying pending entities and relationships with approve/reject actions.

### EPIC07: Production Web Application & Ultra-Fast Graph-RAG
- [x] **T703**: Author unified master system specification (`master_ebv_system_spec.md`) consolidating problem statement, 7-entity core schema, off-the-shelf technology stack (`FastAPI`, `LanceDB`, `KùzuDB`/`Neo4j`, `specter2`, `Cytoscape.js`), and instant web deployment workflow.
- [x] **T704**: Implement MVP operational fixes resolving entity mapping, specter2 embeddings default, built-in biological reference dictionaries, and 3-tier relationship curation status.
- [/] **T705**: Execute 3-tier literature search funnel (Level 1: Organism/Disease, Level 2: Viral Loci/Latency, Level 3: Molecular Interactions) with automated background pueue monitoring.
- [x] **T706**: Implement AnnData `.h5ad` and cluster marker parser (`app/ingestion/anndata_parser.py`) linking single-cell RNA-seq marker genes (`TBX21`, `CXCR3`) directly to `CellState` nodes (`Atypical B Cell`, `GCB`).
- [x] **T707**: Implement Path-to-Text `FactSerializer` (`app/retrieval/fact_serializer.py`) converting retrieved Cypher graph paths into standardized natural language fact triples for LLM prompts.
- [x] **T708**: Implement 2-Hop Subgraph Neighborhood Pruner (`app/retrieval/subgraph_pruner.py`) ranking retrieved graph paths by vector similarity to user prompt.
- [x] **T709**: Implement Unified `GraphRAGPipeline` (`app/retrieval/graph_rag_pipeline.py`) orchestrating LanceDB vector search, multi-hop graph retrieval, path pruning, fact serialization, and LLM synthesis with dual citations.
- [x] **T710**: Implement FastAPI Health & Metrics Router (`app/api/health_routes.py`) serving `/api/v1/health` and `/api/v1/metrics`.

---

## To Do
- None

---

## In Progress
- [/] **T705**: Execute 3-tier literature search funnel (Level 1: Organism/Disease, Level 2: Viral Loci/Latency, Level 3: Molecular Interactions) with automated background pueue monitoring.

---

## Review
- None

---

## Done
- [x] **T304**: Implemented Graph Engine Benchmarking Script (`app/materialization/benchmark_graph_engines.py`) and AgeEngine wrapper (`app/materialization/age_engine.py`) measuring write throughput (nodes/sec, edges/sec) and multi-hop Cypher read latency across Neo4jClient, KuzuEngine, and AgeEngine with full test suite (`tests/test_benchmark_graph_engines.py`).
- [x] **T302**: Implemented `AgeEngine` (`app/materialization/age_engine.py`) wrapping PostgreSQL Apache AGE extension (`ag_catalog`) with Cypher query execution, schema management (`create_graph('ebv_graph')`), bulk node/edge upserts, 2-hop neighborhood path retrieval, and mock fallback (`MockAgeConnection`/`MockAgeDatabase`), verified with pytest suite (`tests/test_age_engine.py`, 10/10 passing).
- [x] **T708**: Implemented 2-Hop Subgraph Neighborhood Pruner (`app/retrieval/subgraph_pruner.py`) ranking retrieved graph paths by vector similarity to user prompt and pruning to top-K for LLM context window optimization, verified with comprehensive unit tests (`tests/test_subgraph_pruner.py`).
- [x] **T504**: Implemented FastAPI Hypothesis Router (`app/api/hypothesis_routes.py`) serving `/api/v1/hypothesis/niche-overlap` querying Neo4j/PostgreSQL for CellState nodes connected to multiple distinct DiseaseOutcome entities across silos, verified with unit tests (`tests/test_hypothesis_routes.py`).
- [x] **T404**: Implemented graph-augmented retriever (`app/retrieval/graph.py`) using Neo4jClient to traverse 2-hop neighborhoods, retrieve related entities/papers, and format context, verified with unit tests (`tests/test_graph_retriever.py`).
- [x] **T403**: Implemented hybrid retriever (`app/retrieval/hybrid.py`) combining dense semantic search (via LanceDB), sparse lexical search (via LanceDB FTS index), and cross-encoder reranking (via sentence-transformers CrossEncoder), verified with unit tests (`tests/test_hybrid.py`).
- [x] **T503**: Implemented embedding evaluation suite (`app/evaluation/benchmark.py`) comparing retrieval metrics (precision@K, recall@K, MRR) across different embedding models on gold-standard EBV queries, verified with comprehensive tests (`tests/test_benchmark.py`).
- [x] **T203**: Implemented entity mapping module (`app/processing/entity_mapper.py`) normalizing raw NER entities to canonical ontology IDs, inserting document metadata, chunks, resolved entities, co-occurring relationships, and citations into PostgreSQL within atomic transactions.
- [x] **T501**: Implemented LLM synthesis client (`app/synthesis/llm.py`, `app/synthesis/prompts.py`) utilizing Claude API to generate factual answers with confidence scoring and citation mapping, verified with comprehensive tests (`tests/test_synthesis.py`).
- [x] **T401**: Implemented LanceDB client wrapper (`app/retrieval/vector.py`) for dense vector schema mapping and idempotent table storage, supporting multiple similarity metrics (L2, Cosine, Dot), and robust ListTablesResponse parsing.
- [x] **T402**: Implemented local BGE-M3 embedding client (`app/retrieval/embeddings.py`) utilizing sentence-transformers, supporting automatic device detection, lazy loading, token sorting optimization, and BGE-M3 sparse lexical weight extraction (with robust fallback).
- [x] **T305**: Implemented the materialization pipeline module in `app/materialization/materializer.py` that reads normalized entities, papers, and relationships from PostgreSQL and writes them into Neo4j with full unit test coverage.
- [x] **T101**: Implemented PMC JATS XML parser (`app/ingestion/pmc_parser.py`) with full multi-level section text chunking, metadata extraction, references bibliography extraction, and pytest suite (`tests/test_pmc_parser.py`).
- [x] **T102**: Implemented Grobid PDF extractor client with fallback to PyMuPDF (`app/ingestion/pdf_extractor.py`) matching JATS schema formats, and test suite (`tests/test_pdf_extractor.py`).
- [x] **T103**: Implemented PubMed API scraper script (`app/ingestion/pubmed_scraper.py`) that queries articles by search terms and downloads PMC XMLs and metadata JSONs to staging with full test coverage (`tests/test_pubmed_scraper.py`).
- [x] **T104**: Implemented GEO/SRA crawler script (`app/ingestion/geo_crawler.py`) to download GSE metadata and series matrices into local staging JSONs with full test suite (`tests/test_geo_crawler.py`).
- [x] **T201**: Implemented SciSpacy and Bern2 API wrapper for NER extraction (`app/processing/ner_extractor.py`) with full unit test suite (`tests/test_ner_extractor.py`).
- [x] **T202**: Implemented local dictionary-based synonym resolver (`app/processing/synonym_resolver.py`) for HGNC, Cell Ontology, DOID, UniProt, and UBERON with fuzzy matching and OLS fallback, and complete test suite (`tests/test_synonym_resolver.py`).
- [x] **T204**: Defined PostgreSQL database schema (`app/database/schema.sql`) and psycopg initialization helper (`app/database/schema.py`) with full unit test coverage.
- [x] **T301**: Implemented Neo4j Graph DB client wrapper (`app/materialization/neo4j_client.py`), constraint and index creation, detach delete, and parameterized Cypher bulk writes, with tests (`tests/test_neo4j_client.py`).
- **T000**: Project agile initialization (setup of Kanban, Handover, Behavior guidelines, and Gemini documentation).
- **T001**: Architecture design alignment and agile documentation setup.
- **T701**: Drafted the detailed "Storage & Database Strategy" in `ebv-rag-engineering-spec.md` specifying relational schemas, vector indexes, and graph rebuild boundaries.
- **T702**: Completed missing "Scaling & Performance", "Security & Data Governance", "Monitoring & Observability", and "Risk & Mitigation" sections in `ebv-rag-engineering-spec.md`..
