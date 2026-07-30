# Agent Handover Document

> **Instructions for Agents:** 
> Update this document at the end of your session if you are handing off a task to another model or suspending work. Ensure the next agent can immediately resume where you left off.

## 1. Current State
- Project configuration initialized and dependencies installed via `uv sync` (`task-69` completed).
- Core specification gaps filled (T701, T702).
- **Core Ingestion Layer Complete**: PMC JATS XML Parser (T101), Grobid PDF Extractor client with PyMuPDF fallback (T102), PubMed API Entrez Scraper (T103), and GEO/SRA Metadata Crawler (T104) are fully implemented and verified.
- **NER & Normalization Core Complete**: SciSpacy and Bern2 API NER Wrapper (T201), Dictionary-based Synonym Resolver (T202), and the Entity Mapping module (T203) are fully implemented.
- **Database Schema Complete**: PostgreSQL schema (T204) and initialization helpers are fully implemented.
- **Neo4j client wrapper Complete**: Neo4j client (T301) for parameterized Cypher writes, unique constraint/index management, and detach delete is fully implemented.
- **Knowledge Graph Materialization Complete**: Materializer module (`app/materialization/materializer.py`) reads normalized entities, papers, and relationships from PostgreSQL and materializes them to Neo4j. MENTIONS relationships are drawn between Papers and Entities. CLI tool and unit tests verified (`T305` complete).
- **Vector Database Client Complete**: LanceDB client wrapper (`app/retrieval/vector.py`) for schema definition, chunk ingestion, and multi-metric vector queries (L2, Cosine, Dot) is fully implemented and tested.
- **Local Embedding Client Complete**: EmbeddingClient (`app/retrieval/embeddings.py`) utilizes sentence-transformers, handles lazy loading and device placement, implements token sorting optimization, and supports BGE-M3 sparse weight extraction with robust fallback. Verification tests in `tests/test_embeddings.py` pass (`T402` complete).
- **Test Coverage**: All 97 unit tests pass successfully.

## 2. Active Tasks (In Progress)
- **T403**: Build hybrid retriever in `app/retrieval/hybrid.py` (dense vector candidates from LanceDB + sparse lexical candidates + cross-encoder reranking).
- **T404**: Implement graph-augmented context retrieval by querying Neo4j multi-hop entity associations.
- **T501**: Design synthesis prompt and utility utilizing Claude API.
- **T503**: Build evaluation test suite comparing specter2 vs BGE-M3.

## 3. Next Steps (Immediate)
- Spawn subagents concurrently to work on T403, T404, T501, and T503.
- Check in every 3-5 minutes to verify progress and run the pytest suite.
- Once completed, build FastAPI server / dashboard and connect everything.

## 4. Pending Blockers or Open Questions
- None.

## 5. Important Context / Gotchas
- The local Python environment is set up under `.venv/` (Python 3.12.10) with `uv` package manager.
- All code must conform strictly to the 12 rules of `AgentBehavior.md`.
- Fail loud, verify with unit tests, and keep final system integrations/linking in mind.
