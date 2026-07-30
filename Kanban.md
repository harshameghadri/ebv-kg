# Project Kanban

This document serves as the agile task tracker for the EBV Knowledge System. It contains atomic, goal-oriented tasks with explicit endpoint-specific definitions.

## Backlog

### EPIC01: Ingestion Layer
- [ ] **T101**: Implement PMC XML parser to extract clean plaintext, metadata, and references directly from JATS xml files.
- [ ] **T102**: Implement Grobid PDF extractor client with fallback to PyMuPDF to extract text from scientific articles into standard JSON formats.
- [ ] **T103**: Implement PubMed API scraper script that queries articles by search terms and downloads metadata and PMC XMLs to the staging directory.
- [ ] **T104**: Implement GEO/SRA crawler script to download GSE metadata and series matrices into local staging JSONs.
- [ ] **T105**: Build a CLI command to ingest and parse single-cell AnnData (`.h5ad`) files and export marker gene CSVs, cell type labels, and spatial coordinates into staging tables.

### EPIC02: Entity Normalization & NER Pipeline
- [ ] **T201**: Set up SciSpacy and Bern2 API wrapper to run NER extraction on parsed plain text, outputting raw entities with confidence scores.
- [ ] **T202**: Implement local dictionary-based synonym resolver for HGNC (genes), Cell Ontology (cells), DOID (diseases), UniProt (proteins), and UBERON (anatomy).
- [ ] **T203**: Implement entity mapping module `app/entity_mapper.py` that normalizes raw NER entities to canonical ontology IDs with combined confidence scores.
- [ ] **T204**: Create PostgreSQL database schema defining raw source data, parsed document chunks, extracted entities, and curation tables.

### EPIC03: Knowledge Graph Materialization
- [ ] **T301**: Define `GraphEngine` protocol in `app/materialization/graph_engine.py` with standard connection, schema initialization, and bulk-load Cypher methods.
- [ ] **T302**: Implement `AgeEngine` using PostgreSQL AGE extension to execute graph operations inside PostgreSQL.
- [ ] **T303**: Implement `LadybugEngine` to execute graph operations on the embedded LadybugDB/Kùzu engine.
- [ ] **T304**: Build a graph engine benchmarking script to measure AGE vs LadybugDB write throughput and multi-hop read latency, outputting a comparative JSON report.
- [ ] **T305**: Implement the materialization pipeline module that reads normalized entities and relationships from PostgreSQL and writes them into the selected graph engine.

### EPIC04: Retrieval & Hybrid Search Layer
- [ ] **T401**: Set up LanceDB vector store client in `app/retrieval/vector.py` supporting Arrow-native dense and metadata indexing.
- [ ] **T402**: Implement local BGE-M3 embedding client in `app/embeddings.py` to generate dense vector embeddings and sparse lexical weights.
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

### EPIC07: Documentation & Spec Gaps
- [ ] **T701**: Draft the detailed "Storage & Database Strategy" in `ebv-rag-engineering-spec.md` to specify the schema designs, indexes, and synchronization boundaries.
- [ ] **T702**: Write the missing "Scaling & Performance", "Security & Data Governance", "Monitoring & Observability", and "Risk & Mitigation" sections in `ebv-rag-engineering-spec.md`.

---

## In Progress
- [ ] **T001**: Architecture design alignment and agile documentation setup.

---

## Review
- None

---

## Done
- **T000**: Project agile initialization (setup of Kanban, Handover, Behavior guidelines, and Gemini documentation).
