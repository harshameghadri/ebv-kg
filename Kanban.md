# Project Kanban

This document serves as the agile task tracker for the EBV Knowledge System. It contains atomic, goal-oriented tasks with explicit endpoint-specific definitions.

## Backlog

### EPIC01: Ingestion Layer
- [ ] **T105**: Build a CLI command to ingest and parse single-cell AnnData (`.h5ad`) files and export marker gene CSVs, cell type labels, and spatial coordinates into staging tables.

### EPIC02: Entity Normalization & NER Pipeline
- [ ] **T203**: Implement entity mapping module `app/entity_mapper.py` that normalizes raw NER entities to canonical ontology IDs with combined confidence scores.

### EPIC03: Knowledge Graph Materialization
- [ ] **T302**: Implement `AgeEngine` using PostgreSQL AGE extension to execute graph operations inside PostgreSQL.
- [ ] **T303**: Implement `LadybugEngine` to execute graph operations on the embedded LadybugDB/Kùzu engine.
- [ ] **T304**: Build a graph engine benchmarking script to measure AGE vs LadybugDB write throughput and multi-hop read latency, outputting a comparative JSON report.
- [x] **T305**: Implement the materialization pipeline module that reads normalized entities and relationships from PostgreSQL and writes them into the selected graph engine.

### EPIC04: Retrieval & Hybrid Search Layer
- [ ] **T401**: Set up LanceDB vector store client in `app/retrieval/vector.py` supporting Arrow-native dense and metadata indexing.
- [x] **T402**: Implement local BGE-M3 embedding client in `app/embeddings.py` to generate dense vector embeddings and sparse lexical weights.
- [ ] **T403**: Build hybrid retriever in `app/retrieval/hybrid.py` that queries LanceDB for dense candidates, combines with sparse lexical candidates, and performs cross-encoder reranking using `bge-reranker-v2`.
- [ ] **T404**: Implement graph-augmented context retrieval by querying multi-hop entity associations from the materialization graph engine to expand the document search candidates.

### EPIC05: LLM Synthesis & RAG Query Layer
- [ ] **T501**: Design synthesis prompt and utility utilizing Claude API to generate factual scientific answers cited against retrieved vector chunks and KG edges.
- [ ] **T502**: Implement FastAPI server with routes for search queries, hybrid semantic retrieval, RAG synthesis, and graph node visualization.
- [ ] **T503**: Build evaluation test suite comparing specter2 vs BGE-M3 embedding models on 50+ gold-standard EBV biological queries, calculating recall@K.

### EPIC06: Human Curation & Discovery Loop
- [ ] **T601**: Implement LightRAG indexing runner in `discovery/lightrag_runner.py` that runs automated clustering and community detection on the PostgreSQL text corpus.
- [ ] **T602**: Build the harvesting script in `discovery/harvest.py` that ranks LightRAG discovery candidates against the canonical KG and promotes the top 20 candidates weekly to the review queue.
- [ ] **T603**: Create web-based Curation Dashboard using HTML/CSS/JS (FastAPI frontend) displaying pending entities and relationships with approve/reject actions.

---

## To Do
- None

---

## In Progress

### EPIC04: Retrieval & Hybrid Search Layer
- [ ] **T403**: Build hybrid retriever in `app/retrieval/hybrid.py` that queries LanceDB for dense candidates, combines with sparse lexical candidates, and performs cross-encoder reranking using `bge-reranker-v2`. *(Assigned to subagent `hybrid-retriever`)*
- [ ] **T404**: Implement graph-augmented context retrieval by querying multi-hop entity associations from the materialization graph engine to expand the document search candidates. *(Assigned to subagent `graph-retriever`)*

### EPIC05: LLM Synthesis & RAG Query Layer
- [ ] **T501**: Design synthesis prompt and utility utilizing Claude API to generate factual scientific answers cited against retrieved vector chunks and KG edges. *(Assigned to subagent `synthesis-engineer`)*
- [ ] **T503**: Build evaluation test suite comparing specter2 vs BGE-M3 embedding models on 50+ gold-standard EBV biological queries, calculating recall@K. *(Assigned to subagent `evaluator`)*

---

## Review
- None

---

## Done
- [x] **T203**: Implemented entity mapping module (`app/processing/entity_mapper.py`) normalizing raw NER entities to canonical ontology IDs, inserting document metadata, chunks, resolved entities, co-occurring relationships, and citations into PostgreSQL within atomic transactions.
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
