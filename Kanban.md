# Project Kanban

This document serves as the agile task tracker for the EBV Knowledge System. It contains an atomic set of goal-oriented tasks based on Scrum/Agile frameworks.

## Backlog

### Epic: Ingestion Layer
- [ ] Set up PDF extraction pipeline using Grobid/pymupdf
- [ ] Implement PMC XML primary ingestion pipeline
- [ ] Implement PubMed API fallback scraper
- [ ] Build GEO/SRA metadata crawler
- [ ] Index portfolio QMDs

### Epic: Data Processing Pipeline
- [ ] Implement text extraction and quality control validation
- [ ] Integrate scispacy and Bern2 API for Entity Extraction & NER
- [ ] Set up entity synonym normalization
- [ ] Develop LLM-based relationship inference module
- [ ] Implement confidence scoring gate (0.5+ threshold)

### Epic: Human Curation
- [ ] Create UI queue for human curation
- [ ] Build entity approve/reject functionality
- [ ] Build relationship validation interface

### Epic: Materialization & Storage
- [ ] Provision PostgreSQL for raw data and state
- [ ] Provision Neo4j for Knowledge Graph storage
- [ ] Set up ChromaDB for vector embeddings

### Epic: API & Query Layer
- [ ] Design GraphQL/REST endpoints for KG queries
- [ ] Implement semantic search endpoints
- [ ] Add rate limiting and caching

## To Do
- [ ] Initialize repository structure and linting
- [ ] Define shared data schemas (Pydantic/JSON Schema)

## In Progress
- [ ] Architecture design and documentation

## Review
- [ ] None

## Done
- [ ] Initial project specification and engineering documents
- [ ] Agile documentation setup (Kanban, Handover, Behavior guidelines)
