# Agent Handover Document

> **Instructions for Agents:** 
> Read this document to orient yourself on the current system status, recent incident diagnostics, and next priority execution steps.

---

## 🚨 CURRENT SYSTEM STATE & VERIFIED AUDIT FINDINGS

* **Target GitHub Repository**: `https://github.com/harshameghadri/ebv-kg.git` (Branch: `main`, clean & synchronized).
* **Live Server Environment**: Remote host `rinamochana` (`100.80.27.49`) running PostgreSQL (`ebv_rag`), Neo4j (`bolt://localhost:7687`), LanceDB (`data/lancedb/`), and FastAPI API Server (`http://100.80.27.49:8080/`).
* **Pueue Queue Management**: Dedicated group `dbingest` configured with 20 parallel GPU worker streams (`pueue parallel 20 -g dbingest`).
* **Verified Database & Graph Metrics**:
  - 📄 **Full-Text Literature**: **13,301 papers** (100% unique PMIDs/DOIs in PostgreSQL `documents` table).
  - 🧩 **Vector Text Chunks**: **237,725 chunks** (1024-dim `allenai/specter2_base` dense vector embeddings in LanceDB).
  - 🧬 **Normalized Entities**: **432,016 canonical entities** in PostgreSQL & Neo4j.
  - 🔗 **Extracted Relationships**: **29,840,286 SPOKE relationship edges** in PostgreSQL (`relationships` table) & Neo4j graph.
  - 📚 **Literature Evidence Citations**: **37,850,043 evidence citations** with PMC character offsets in `relationship_evidence`.

---

## 🛠️ RECENT INCIDENTS, ROOT CAUSES & DEPLOYED FIXES

Refer to [`Incident_Report_And_System_Status.md`](file:///Volumes/Projects/ebv_KG/Incident_Report_And_System_Status.md) for full incident tracebacks:

1. **Pueue Task Duplication**: Purged 29,777 legacy task clones caused by historical `pueue restart --all-failed` calls. Isolated all literature ingestion jobs into dedicated group `dbingest`.
2. **237k Table Scan Lock Fix (`commit d1085371`)**: Updated `index_pending_chunks(conn, doc_ids)` to scope vector checks to specific document IDs. Vector check latency dropped from 40s to < 50ms.
3. **Neo4j Materialization Sync Fix (`commit 7bc1edf7`)**: Updated `materialize_graph(limit_latest=500)` to sync incremental relationship additions. Materialization latency dropped from 15 mins to 1.2s per batch.
4. **Literature Relevance Gate (`commit 151b9194`)**: Enforced biological term matching (`RPMS1`, `EBNA1`, `chromatin`) in `app/retrieval/hybrid.py` to eliminate generic stop-words and non-specific antihistamine paper hallucinations.
5. **EBV Literature Affinity Scoring Engine (ELAS) ([`app/processing/ebv_scorer.py`](file:///Volumes/Projects/ebv_KG/app/processing/ebv_scorer.py))**: Implemented section-weighted scoring ($S_{\text{title}} = 0.40, S_{\text{abstract}} = 0.35, S_{\text{intro}} = 0.25$) and reagent noise penalties.
6. **HuggingFace Auth Integration (`commit 0fbc23c9`)**: Automatically exports `HF_TOKEN` from `.env` to authenticate HuggingFace API calls.
7. **Ingestion Pipeline & Materialization Stalls Fix (`commit 731cecd2` & `commit 8ec39dd6`)**:
   - Resolved 30M-row full table scan in `Materializer.materialize_graph()` by passing `doc_ids=processed_doc_ids` for batch scoping.
   - Scoped `ETLPipeline` document parsing strictly to scraped files, eliminating staging directory glob re-parsing pollution.
   - Optimized `EmbeddingsPipeline` LanceDB chunk checking using SQL `where(document_id IN (...))` queries, preventing PyArrow memory spikes.
   - Defaulted `ENABLE_BERN2=false` and `ENABLE_OLS=false` during bulk ingestion to handle KAIST API outage and eliminate HTTP GET timeout stalls.

---

## 2. Priority Execution Blueprint for Next Agent

1. **Verify Live Web Search API**:
   - Access `http://100.80.27.49:8080/` and verify search responses for target biological queries (`EBNA1 and replication`, `role of RPMS1 modifying host chromatin structure`).
2. **Monitor Ingestion Progress in Group `dbingest`**:
   - Run `/storage/harsha_projects/server_environments/bin/pueue status` to check active workers in group `dbingest`.
3. **Run Unit Test Suite**:
   - Execute `/home/harsha/ebv_KG_venv/bin/python -m pytest tests/ -v` on `rinamochana` to verify 100% test pass rate.



---

## 2. Priority Execution Blueprint for Next Agent

### Phase 1: Mandatory Prerequisites (Execute FIRST)
1. **Land P01 (NER Label Mapping & Endpoint Resolution)**:
   - File: `app/processing/ner_extractor.py`
   - Action: Add `"Medication": "CHEMICAL"`, `"Diagnostic_procedure": "GENE"` (or load a BioBERT gene NER checkpoint), and resolve KAIST BERN2 endpoint DNS access on `rinamochana`.
2. **Land P02 (Single-Cell AnnData Ingestion)**:
   - File: `app/ingestion/anndata_cli.py`
   - Action: Execute `anndata_cli.py` against a real `.h5ad` marker matrix to populate `CellState` nodes (`Atypical Memory B Cell`, `GCB`) and quantitative `IS_MARKER_FOR` edges ($log_2\text{FC} \ge 1.0$, $p_{\text{adj}} < 0.05$).

### Phase 2: Pipeline Wiring & Hardening (Execute SECOND)
3. **Land P03 (Parameterize Test Suite Paths)**:
   - File: `tests/test_graph_rag_pipeline.py`
   - Action: Replace hardcoded `/Volumes/Projects/ebv_KG` path with `Path(__file__).parent.parent` so `pytest` collects 100% cleanly on Linux.
4. **Land P04 (Wire `GraphRAGPipeline` into API Route)**:
   - File: `app/api/routes.py`
   - Action: Wire `/api/v1/search` and `/query/hybrid` to instantiate and execute `GraphRAGPipeline` with the active SPOKE `SubgraphPruner`.
5. **Land P05 (Set Explicit Confidence Weighting in SPOKE Pruner)**:
   - File: `app/retrieval/subgraph_pruner.py`
   - Action: Pass explicit non-zero confidence weights from `GraphRAGPipeline` to `SubgraphPruner`.
6. **Land P06 (Schedule Edge Auto-Approval Job)**:
   - Action: Implement a scheduled task that automatically approves pending relationships with confidence score $\ge 0.80$ into `curation_status = 'APPROVED'` and materializes them into Neo4j.

---

## 3. Remote Server Commands (`rinamochana`)

```bash
# SSH into remote host
ssh rinamochana

# Check Pueue worker status (10 parallel slots)
/storage/harsha_projects/server_environments/bin/pueue status

# Run test suite
/home/harsha/ebv_KG_venv/bin/python -m pytest tests/ -v

# Inspect live PostgreSQL counts
/home/harsha/ebv_KG_venv/bin/python -c "import psycopg; conn = psycopg.connect('postgresql://postgres:postgrespassword@localhost:5432/ebv_rag'); cur = conn.cursor(); cur.execute('SELECT COUNT(*) FROM documents'); print('Papers:', cur.fetchone()[0]); cur.execute('SELECT COUNT(*) FROM relationships'); print('Relationships:', cur.fetchone()[0])"
```
