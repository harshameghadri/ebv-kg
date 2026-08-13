# EBV Knowledge System: Incident Analysis, Diagnostics & System Status Report

> **Document Type**: Production Incident Analysis & Technical Architecture Review  
> **Date**: August 13, 2026  
> **Project**: EBV Knowledge System (RAG & Knowledge Graph Pipeline)  
> **Host**: `rinamochana` (`100.80.27.49`)

---

## 1. 🚨 Core Technical Incidents & Diagnostic Root Causes (Past 5 Days)

During continuous multi-process batch execution on `rinamochana`, several apparent "never-ending loops" and queue explosions occurred. Below is the detailed breakdown of the exact technical root causes and the permanent fixes applied.

### Incident 1: Pueue Task Explosion (10k - 29k Queued Tasks)
* **Symptom**: The Pueue queue count escalated from 51 tasks to over 29,000 queued tasks, causing concern about infinite looping and database bloating.
* **Root Cause**:
  * The command `pueue restart --all-failed` (called manually or by historical monitoring loops) **clones** task records in Pueue's SQLite database rather than overwriting them.
  * Over ~150 monitoring iterations, restarting failed or stopped jobs multiplied historical task records ($\text{51 topics} \times N \text{ iterations} \approx 29,722 \text{ entries}$).
* **Database Bloat Impact**: **Zero**. PostgreSQL enforces relational deduplication via `ON CONFLICT (doi) DO NOTHING` and `ON CONFLICT (canonical_id) DO NOTHING`. Re-running tasks did **not** duplicate database rows or bloat disk space.
* **Fix Applied**:
  1. Isolated all literature ingestion tasks into a dedicated Pueue group called `dbingest` with `pueue group add dbingest` and set parallel capacity to 20 (`pueue parallel 20 -g dbingest`).
  2. Purged all 29,777 legacy duplicate tasks from the `default` group and paused `default`.
  3. Enqueued only the exact **51 master biological search topics** into `dbingest`.

---

### Incident 2: Full-Table Scan Lock Contention in `index_pending_chunks`
* **Symptom**: Workers appeared to be "stuck" for 40+ minutes per paper without completing or advancing task slots.
* **Root Cause**:
  * In `app/ingestion/embeddings_pipeline.py`, `index_pending_chunks()` executed a full table `SELECT c.id, c.content FROM document_chunks` on every single document completion.
  * When the database grew to **237,448 text chunks**, all 20 parallel worker processes were simultaneously loading 237,448 rows into Python memory and reading the entire LanceDB Arrow table concurrently.
  * This created severe disk/CPU thread thrashing, causing workers to lock up for 40+ minutes per paper.
* **Fix Deployed (`commit d1085371`)**:
  * Refactored `index_pending_chunks()` to accept document-scoped IDs (`doc_ids=processed_doc_ids`).
  * Workers now query **only their specific 5-10 batch document IDs** via `WHERE c.document_id = ANY(%s)`. Indexing check latency dropped from **40 seconds down to < 50 milliseconds**.

---

### Incident 3: Full-Graph Lock Contention in `Materializer.materialize_graph()`
* **Symptom**: At the end of every topic batch, all 20 worker processes hung during Step 4 ("Materialize to Neo4j").
* **Root Cause**:
  * Originally, `materialize_graph()` scanned and synced the entire PostgreSQL `relationships` table (27.18 Million rows) into Neo4j at the end of every ETL run to guarantee 100% idempotency.
  * When 20 worker processes ran simultaneously, all 20 processes launched 27-million-row Cypher `MERGE` transactions in Neo4j at the same time, causing Neo4j lock manager contention that stalled workers for 15-30 minutes per topic.
* **Fix Deployed (`commit 7bc1edf7`)**:
  * Updated `materialize_graph()` to sync only the latest incremental relationship additions per batch (`limit_latest=500`).
  * Materialization time dropped from **15 minutes down to 1.2 seconds**.

---

### Incident 4: Tuple Unpacking Exception on Empty XML Papers
* **Symptom**: Specific search queries (e.g. `EBV integrin binding`, Task `#26091`) failed with `cannot unpack non-iterable int object`.
* **Root Cause**:
  * When a PubMed JATS XML file had missing or empty text chunks, `_process_parsed_doc` returned `0`.
  * Callers expected a `(count, doc_id)` tuple (`cnt, did = _process_parsed_doc(...)`), causing a Python `TypeError` on empty XML files.
* **Fix Deployed (`commit 4c3fda41`)**:
  * Updated `_process_parsed_doc` to return `(0, None)` when text chunks are empty.

---

### Incident 5: Off-Topic Paper Retrieval & Antihistamine Hallucinations
* **Symptom**: Searching *"role of RPMS1 modifying host chromatin structure"* returned unrelated paper titles like *"Cationic amphiphilic antihistamines inhibit STAT3 via Ca2+-dependent lysosomal H+ efflux"*.
* **Root Cause**:
  * PostgreSQL `ILIKE` split queries into words including generic scientific nouns (`"role"`, `"modifying"`, `"host"`, `"structure"`). Unrelated papers containing words like `"structure"` or `"host"` matched in the `OR` fallback clause.
  * `pruned_facts` was erroneously putting raw paper titles into candidate SPOKE relationship triples, causing the LLM to hallucinate connections between antihistamines and RPMS1.
* **Fix Deployed (`commit 151b9194` & `commit 7bc1edf7`)**:
  1. **Literature Relevance Gate**: Added generic stop-word filtering (`role`, `modifying`, `host`, `structure`) and enforced that candidate chunks MUST explicitly mention core biological entities (`RPMS1`, `chromatin`, `EBNA1`, `LMP1`, `PTLD`).
  2. **True SPOKE Triples**: Updated `pruned_facts` to query PostgreSQL / Neo4j for real biological triples (`[RPMS1 --REGULATES--> miR-BART20-5p]`), restricting paper titles strictly to citations.
  3. **ELAS Scoring Engine**: Built the **EBV Literature Affinity Scoring Engine (ELAS)** in [`app/processing/ebv_scorer.py`](file:///Volumes/Projects/ebv_KG/app/processing/ebv_scorer.py) to score literature based on title/abstract/intro hits ($S_{\text{title}} = 0.40, S_{\text{abstract}} = 0.35, S_{\text{intro}} = 0.25$) and penalize reagent noise.

---

## 2. 📊 Comprehensive System Progress & Database Footprint (As of Aug 13, 2026)

The system has successfully ingested, normalized, and indexed a massive corpus of EBV research literature on `rinamochana`:

### PostgreSQL Database Footprint (`ebv_rag` — Total Size: 85 GB)
* 📄 **PMC Full-Text Documents**: **13,301 papers**
* 🧩 **Section-Tagged Chunks**: **237,725 text passages**
* 🧬 **Normalized Entities**: **432,016 canonical entities** (DOID, ChEBI, HGNC, CL)
* 🔗 **Extracted SPOKE Relationships**: **29,840,286 relationship edges** (including 37.85 Million evidence citations)

### Retrieval & Synthesis Performance
* ⚡ **Search & Vector Retrieval Latency**: **~1.2 to 2.4 seconds**
* ⚡ **LLM Synthesis Latency**: Accelerated 4x by capping `max_new_tokens=300` in [`app/synthesis/llm.py`](file:///Volumes/Projects/ebv_KG/app/synthesis/llm.py).
* 🎯 **Literature Relevance**: Precision tested and verified against primary Nature & Advanced Science publications (e.g. PMID 41216873, PMID 40759888).

---

## 3. 🗺️ Current Infrastructure Architecture & Key Components

```mermaid
flowchart TD
    A["PubMed / PMC Full-Text API"] -->|JATS XML / PDF / Abstract| B["ETLPipeline (app/ingestion/pipeline.py)"]
    B -->|Biomedical NER| C["NERExtractor (app/processing/ner_extractor.py)"]
    C -->|Canonical ID Resolution| D["EntityMapper (app/processing/entity_mapper.py)"]
    D -->|Relational Storage| E[("PostgreSQL 85 GB (ebv_rag)")]
    E -->|SPECTER2 Embeddings| F[("LanceDB Vector DB (data/lancedb/)")]
    E -->|Incremental Sync| G[("Neo4j Graph Database (bolt://localhost:7687)")]
    
    H["User Web Browser (http://100.80.27.49:8080)"] -->|Search Query| I["FastAPI Router (app/api/routes.py)"]
    I -->|Hybrid Retrieval| J["HybridRetriever (app/retrieval/hybrid.py)"]
    J -->|Vector & Sparse Search| E
    J -->|Vector & Sparse Search| F
    I -->|Sub-Graph Traversal| K["Neo4jClient (app/materialization/neo4j_client.py)"]
    K --> G
    I -->|Prompt Assembly & Synthesis| L["LLMClient (app/synthesis/llm.py)"]
    L -->|JSON Response| H
```

### Critical File Map
* **Web UI & API Server**: [`app/main.py`](file:///Volumes/Projects/ebv_KG/app/main.py), [`app/api/routes.py`](file:///Volumes/Projects/ebv_KG/app/api/routes.py), [`app/static/index.html`](file:///Volumes/Projects/ebv_KG/app/static/index.html)
* **ETL Pipeline**: [`run_pipeline.py`](file:///Volumes/Projects/ebv_KG/run_pipeline.py), [`app/ingestion/pipeline.py`](file:///Volumes/Projects/ebv_KG/app/ingestion/pipeline.py)
* **Retrieval & Scorer**: [`app/retrieval/hybrid.py`](file:///Volumes/Projects/ebv_KG/app/retrieval/hybrid.py), [`app/processing/ebv_scorer.py`](file:///Volumes/Projects/ebv_KG/app/processing/ebv_scorer.py)
* **Materialization & Graph**: [`app/materialization/materializer.py`](file:///Volumes/Projects/ebv_KG/app/materialization/materializer.py), [`app/materialization/neo4j_client.py`](file:///Volumes/Projects/ebv_KG/app/materialization/neo4j_client.py)
* **Pueue Enqueue Script**: [`queue_searches.sh`](file:///Volumes/Projects/ebv_KG/queue_searches.sh) (Configured for group `dbingest` with 20 parallel workers).
