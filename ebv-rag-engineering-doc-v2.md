# EBV Knowledge System: Engineering & Infrastructure Specification
## Version 2.0 (Critique-Integrated)

**Status**: Design Phase (Pre-Implementation)  
**Last Updated**: 2025-05-04  
**Version**: 2.0 (Incorporates Senior Engineer Critique)  
**Owner**: Harsha (Bioinformatician + Full-Stack Developer)  
**Document Type**: Technical Architecture & DevOps Specification

---

## Executive Summary

This document specifies a **production-grade RAG + Knowledge Graph system for EBV research**, addressing both biomedical domain complexity and distributed systems realities. **Key change from v1**: Emphasis on human-in-the-loop curation, rigorous confidence scoring, and state consistency across databases rather than naive automation.

**Critical Design Principle:** The MVP is not about a "fully automated KG." It's about building **infrastructure to manage a growing, curated knowledge graph** where you validate early relationships manually, then use those to train automation.

---

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Critical Biology & Data Science Design Gaps](#critical-biology--data-science-design-gaps)
3. [Infrastructure & Resource Requirements](#infrastructure--resource-requirements)
4. [Data Pipeline & ETL (Revised)](#data-pipeline--etl-revised)
5. [Storage & Database Strategy](#storage--database-strategy)
6. [Knowledge Graph Design & Confidence Scoring](#knowledge-graph-design--confidence-scoring)
7. [Entity Normalization & Curation](#entity-normalization--curation)
8. [API & Query Layer Design](#api--query-layer-design)
9. [Deployment & DevOps](#deployment--devops)
10. [Testing & Evaluation](#testing--evaluation)
11. [Cost Analysis](#cost-analysis)
12. [Scaling & Performance](#scaling--performance)
13. [Security & Data Governance](#security--data-governance)
14. [Monitoring & Observability](#monitoring--observability)
15. [Risk & Mitigation](#risk--mitigation)
16. [Known Gaps & Phase 1 Validation Tasks](#known-gaps--phase-1-validation-tasks)
17. [Version Control & Changelog](#version-control--changelog)

---

## System Architecture Overview

### High-Level Data Flow (REVISED)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INGESTION LAYER                             │
├─────────────────────────────────────────────────────────────────────┤
│  PDF Extraction  │  PMC XML   │  PubMed API  │  GEO/SRA  │Portfolio │
│  (Grobid/pymupdf)│  (primary) │  (fallback)  │  Crawler  │ QMDs     │
└──────────┬────────────────────────────────────────────────┬──────────┘
           │                                                │
           ▼                                                ▼
┌─────────────────────────────────┐     ┌──────────────────────────┐
│     DATA PROCESSING PIPELINE    │     │  PORTFOLIO INTEGRATION   │
│  ┌──────────────────────────┐   │     │  ┌────────────────────┐  │
│  │ Text Extraction & QC     │   │     │  │ Parse & Index:     │  │
│  │ • Validate text quality  │   │     │  │ • AnnData (.h5ad)  │  │
│  │ • Remove junk/boilerplate│   │     │  │ • Marker genes CSV │  │
│  │ • Deduplicate by DOI     │   │     │  │ • Cell type labels │  │
│  └──────────────────────────┘   │     │  │ • Spatial coords   │  │
│  ┌──────────────────────────┐   │     │  └────────────────────┘  │
│  │ Entity Extraction & NER  │   │     └──────────────────────────┘
│  │ • scispacy + Bern2 API   │   │
│  │ • Synonym normalization  │   │
│  │ • Confidence scoring     │   │
│  └──────────────────────────┘   │
│  ┌──────────────────────────┐   │
│  │ Relationship Inference   │   │
│  │ • LLM-based extraction   │   │
│  │ • Co-occurrence analysis │   │
│  │ • Confidence gate (0.5+) │   │
│  │ • FLAG FOR REVIEW        │   │
│  └──────────────────────────┘   │
│  ┌──────────────────────────┐   │
│  │ Human Curation Queue     │   │
│  │ • Approve/reject entities│   │
│  │ • Validate relationships │   │
│  │ • Refine confidence      │   │
│  └──────────────────────────┘   │
└─────────────────┬─────────────────┘
                  │
      ┌───────────▼────────────────────────┐
      │  MATERIALIZATION LAYER             │
      │  (PostgreSQL → Neo4j/ChromaDB)     │
      │  • Idempotent rebuild nightly      │
      │  • Rollback-safe transactions      │
      │  • Eventual consistency model      │
      └───────────┬────────────────────────┘
                  │
      ┌───────────▼────────────────────────┐
      │ KNOWLEDGE REPRESENTATION LAYER      │
      ├────────────────────────────────────┤
      │ PostgreSQL (Source of Truth)       │
      │ ├─ Papers, chunks, entities        │
      │ ├─ Relationships (curated)         │
      │ ├─ Audit trail & curation logs     │
      │ └─ Pending review queue            │
      │                                     │
      │ Neo4j (Materialized View)          │
      │ ├─ Graph structure only            │
      │ ├─ Rebuilt nightly from PG         │
      │ └─ Deletable & reconstructable     │
      │                                     │
      │ ChromaDB (Materialized View)       │
      │ ├─ Document embeddings (specter2)  │
      │ ├─ Rebuilt with new chunks         │
      │ └─ Deletable & reconstructable     │
      └───────────┬────────────────────────┘
                  │
      ┌───────────▼──────────────────┐
      │  QUERY & SYNTHESIS LAYER     │
      ├──────────────────────────────┤
      │  Hybrid Search Engine        │
      │  • SQLite FTS5 (keyword)     │
      │  • Vector similarity (specter2)
      │  • Cypher graph traversal    │
      │  • Confidence-filtered       │
      │                              │
      │  LLM Chain (Claude)          │
      │  • Retrieval ranking         │
      │  • Multi-hop reasoning       │
      │  • Citation extraction       │
      │  • Confidence scoring        │
      └───────────┬────────────────────┘
                  │
      ┌───────────▼──────────────┐
      │   INTERFACE LAYER        │
      ├───────────────────────────┤
      │ Streamlit Web UI          │
      │ • Query + explore         │
      │ • Curation dashboard      │
      │ • Portfolio integration   │
      │                           │
      │ REST API (FastAPI)        │
      │ • Programmatic access     │
      │ • Webhooks                │
      └───────────────────────────┘
```

**Key Change from v1:** Explicit human curation gate before relationships enter Neo4j. PostgreSQL is the **single source of truth**; Neo4j and ChromaDB are **reconstructable materialized views**.

---

## Critical Biology & Data Science Design Gaps

### Issue 1: The Seurat/R to Python Chasm

**Problem (v1):** Engineering doc naively assumes you can parse raw `.rds` Seurat objects on-the-fly in ETL.

**Reality:**
- RDS serialization is brittle across R versions
- Complex slot structures (scale.data, reductions, metadata) don't serialize cleanly to JSON
- Scheduled ETL pipelines should **never** depend on live R processes

**Solution:**
Mandate that upstream bioinformatics pipelines export to **standardized formats**:

```r
# In your portfolio analysis pipeline (R)

# Export markers (gold standard)
write.csv(
  all_markers,
  file = "outputs/P1_markers_annotated.csv",
  row.names = FALSE
)

# Export normalized counts in AnnData-compatible format
sce <- as.SingleCellExperiment(seurat_obj)
write_h5ad(convertTo(sce, "sce"), "outputs/P1_counts.h5ad")

# Export metadata (cell type assignments, QC metrics)
write.csv(
  seurat_obj@meta.data,
  file = "outputs/P1_metadata.csv"
)

# Export spatial coordinates (for spatial projects)
write.csv(
  seurat_obj@images$slice1@coordinates,
  file = "outputs/P1_spatial_coords.csv"
)
```

**Python ingestion (clean):**

```python
import pandas as pd
import anndata as ad

# 1. Load markers
markers_df = pd.read_csv("P1_markers_annotated.csv")
# Guaranteed structure: cluster, gene, p_val, avg_logFC, pct.1, pct.2

# 2. Load count matrix
adata = ad.read_h5ad("P1_counts.h5ad")

# 3. Load metadata
metadata_df = pd.read_csv("P1_metadata.csv", index_col=0)

# Proceed with normalized, validated data
```

**Implementation:**
- Create an `outputs/` manifest schema (required files, formats, checksums)
- Validate each export in your R analysis
- Store AnnData objects directly in S3/MinIO (queryable via Scanpy)

**Timeline Impact:** +2 days for standardization, saves 2 weeks of integration debugging.

---

### Issue 2: Entity Normalization & Synonymy (CRITICAL)

**Problem (v1):** scispacy extracts entities, but no canonical mapping.

**Example Nightmare:**
```
Paper A: "EBNA-1 expression in LCLs"
Paper B: "Epstein-Barr nuclear antigen 1 drives transformation"
Paper C: "EBNA1 is essential for..."

Current system: 3 separate entity nodes (EBNA-1, EBNA1, "Epstein-Barr nuclear antigen 1")
Reality: Should be ONE node with 3 aliases
```

**Solution: Multi-Step Normalization**

```python
# Step 1: NER extraction (scispacy)
entities_raw = ner_model.predict("EBNA-1 drives B cell transformation")
# Output: [("EBNA-1", PROTEIN, 0.91), ("B cell", CELL_TYPE, 0.87)]

# Step 2: Canonical linking (Bern2 API or local lookup)
from entity_mapper import EntityNormalizer

normalizer = EntityNormalizer(
    ontologies={
        "GENE": load_hgnc(),         # HGNC gene symbols
        "PROTEIN": load_uniprot(),   # UniProt canonical IDs
        "CELL_TYPE": load_cell_ontology(),  # CL ontology
        "DISEASE": load_doid()       # Disease Ontology
    }
)

canonical_entities = []
for entity_text, entity_type, confidence in entities_raw:
    result = normalizer.resolve(entity_text, entity_type)
    # result = {
    #     "canonical_id": "ENSG00000213281",  # HGNC ID
    #     "symbol": "EBNA1",
    #     "aliases": ["EBNA-1", "Epstein-Barr nuclear antigen 1"],
    #     "db_xref": {"HGNC": "HGNC:3236", "Uniprot": "P03211"},
    #     "confidence": 0.95,
    #     "source": "hgnc"
    # }
    canonical_entities.append(result)

# Step 3: Store in PostgreSQL with normalization metadata
for entity in canonical_entities:
    db.upsert_entity(
        canonical_id=entity["canonical_id"],
        entity_type=entity_type,
        symbol=entity["symbol"],
        aliases=entity["aliases"],
        db_xrefs=entity["db_xref"],
        confidence=entity["confidence"],
        source=entity["source"],
        last_seen=now()
    )
```

**Infrastructure:**

```python
# app/entity_mapper.py
class EntityNormalizer:
    def __init__(self, ontologies):
        self.hgnc = ontologies["GENE"]
        self.cl = ontologies["CELL_TYPE"]
        self.doid = ontologies["DISEASE"]
        # Local cache of known mappings
        self.alias_cache = load_alias_db()
    
    def resolve(self, entity_text, entity_type):
        """
        Resolve raw text to canonical entity
        Priority: 1) Alias cache, 2) Fuzzy match to ontology, 3) Bern2 API
        """
        # Check cache first (95% of cases)
        if entity_text in self.alias_cache:
            return self.alias_cache[entity_text]
        
        # Try fuzzy match to ontology
        if entity_type == "GENE":
            match = self.hgnc.fuzzy_match(entity_text, threshold=0.8)
            if match:
                return match
        
        # Fall back to Bern2 (if available)
        bern_result = self.bern2_api.query(entity_text, entity_type)
        if bern_result.confidence > 0.7:
            return bern_result
        
        # If all else fails, create orphan entity (flag for review)
        return {
            "canonical_id": f"unknown_{hash(entity_text)}",
            "symbol": entity_text,
            "aliases": [],
            "confidence": 0.3,
            "source": "unresolved",
            "needs_curation": True
        }

# Load ontologies at startup
def load_hgnc():
    df = pd.read_csv("hgnc_complete_set.csv")
    # Create lookup: {symbol: {ensembl_id, aliases, ...}}
    return df.set_index("symbol").to_dict("index")

def load_cell_ontology():
    # Load from OBO format or JSON
    return CellOntology("cl.obo")

def load_doid():
    # Load Disease Ontology
    return DiseaseOntology("doid.obo")
```

**Data Sources (Free):**
- **HGNC** (genes): `ftp://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/` (updated monthly)
- **UniProt** (proteins): `https://www.uniprot.org/downloads` (weekly updates)
- **Cell Ontology**: `https://github.com/obophenotype/cell-ontology` (GitHub)
- **DOID** (diseases): `https://github.com/DiseaseOntology/HumanDiseaseOntology` (GitHub)
- **Bern2 API** (if needed): Free tier available, rate-limited

**Timeline Impact:** +3 days for ontology integration, but **critical for KG quality**.

---

### Issue 3: Embedding Model Choice (Biomedical Tuning)

**Problem (v1):** all-MiniLM-L6-v2 is general-purpose; performs poorly on biomedical synonymy.

**Test Case:**
```python
from sentence_transformers import SentenceTransformer
import numpy as np

model_general = SentenceTransformer("all-MiniLM-L6-v2")
model_biomedical = SentenceTransformer("allenai/specter2")

# Query: "What is the role of atypical B cells in EBV infection?"
# Documents:
# Doc A: "TBX21+ CXCR3+ B cells (ABCs) are present in EBV+ individuals"
# Doc B: "Atypical B cells express inhibitory checkpoints"
# Doc C: "BCR signaling triggers normal B cell differentiation"

query = "atypical B cells"
doc_a = "TBX21+ CXCR3+ B cells (ABCs) are present in EBV+ individuals"
doc_c = "BCR signaling triggers normal B cell differentiation"

# With all-MiniLM:
emb_q_general = model_general.encode(query)
emb_a_general = model_general.encode(doc_a)
emb_c_general = model_general.encode(doc_c)
score_a = np.dot(emb_q_general, emb_a_general)  # ~0.65
score_c = np.dot(emb_q_general, emb_c_general)  # ~0.62 (too close!)

# With specter2:
emb_q_bio = model_biomedical.encode(query)
emb_a_bio = model_biomedical.encode(doc_a)
emb_c_bio = model_biomedical.encode(doc_c)
score_a = np.dot(emb_q_bio, emb_a_bio)  # ~0.89
score_c = np.dot(emb_q_bio, emb_c_bio)  # ~0.41 (correctly separated)
```

**Solution:**

Use **specter2** (allenai/specter2) for MVP:
- Tuned on biomedical literature
- Drop-in replacement for sentence-transformers
- 768-dim (reasonable size)
- Open-source, maintained by AllenAI

```python
# app/embeddings.py
from sentence_transformers import SentenceTransformer

# MVP: Local biomedical model
EMBEDDING_MODEL = "allenai/specter2"
model = SentenceTransformer(EMBEDDING_MODEL)

# Production: Claude API (better quality, costs scale)
USE_CLAUDE_EMBEDDINGS = True  # Feature flag
CLAUDE_EMBEDDING_DIM = 1024

async def embed_chunk(text: str) -> list[float]:
    if USE_CLAUDE_EMBEDDINGS:
        response = await anthropic_client.embeddings.create(
            input=text,
            model="text-embedding-3-large"
        )
        return response.data[0].embedding
    else:
        # Local fallback
        return model.encode(text, convert_to_numpy=True).tolist()
```

**Cost Comparison:**
- specter2 (local): $0 (once downloaded)
- Claude embeddings: $0.02 per million tokens → ~$30/month at production scale

**Timeline:** No extra time; just swap the model name.

---

### Issue 4: PDF Extraction Strategy (Practical Optimization)

**Problem (v1):** Assumes pymupdf + paddleocr handles all cases; complex layouts fail.

**Solution: Hierarchical Extraction**

```python
# Step 1: Try PMC XML (60% of recent papers)
# Step 2: Fall back to Grobid (structure + citations)
# Step 3: Fall back to pymupdf + OCR (simple cases)

async def extract_paper_content(paper: Paper):
    """
    Hierarchical extraction: PMC XML → Grobid → pymupdf
    """
    
    # Step 1: Check PMC XML availability
    pmc_id = get_pmc_id_from_doi(paper.doi)
    if pmc_id:
        try:
            xml_content = fetch_pmc_xml(pmc_id)
            text, tables, figures = parse_pmc_xml(xml_content)
            return {
                "text": text,
                "tables": tables,
                "figures": figures,
                "source": "pmc_xml",
                "quality": 0.95
            }
        except Exception as e:
            logger.warning(f"PMC XML fetch failed for {paper.doi}: {e}")
    
    # Step 2: Try Grobid (if PDF available)
    if paper.pdf_url:
        try:
            grobid_output = await grobid_client.process_pdf(
                paper.pdf_url,
                timeout=10  # seconds
            )
            # Grobid returns structured TEI XML
            text, tables, citations = parse_grobid_tei(grobid_output)
            return {
                "text": text,
                "tables": tables,
                "citations": citations,
                "source": "grobid",
                "quality": 0.85
            }
        except Exception as e:
            logger.warning(f"Grobid processing failed for {paper.doi}: {e}")
    
    # Step 3: Fall back to pymupdf + OCR
    try:
        text, tables = extract_pymupdf_with_ocr(paper.pdf_url)
        return {
            "text": text,
            "tables": tables,
            "source": "pymupdf_ocr",
            "quality": 0.60  # Lower confidence
        }
    except Exception as e:
        logger.error(f"All extraction methods failed for {paper.doi}: {e}")
        return {"text": "", "error": str(e), "quality": 0}
```

**Infrastructure (Optional Grobid):**

```yaml
# docker-compose.yml
services:
  grobid:
    image: grobid/grobid:0.8.0
    ports:
      - "8070:8070"
    environment:
      - JAVA_OPTS=-Xmx4g  # 4GB heap
    # Note: This is optional; skip for MVP if simplicity preferred
```

**Practical Decision:**
- **Week 1-4 (MVP):** Use Grobid optional; implement fallback to pymupdf
- **Week 5-8:** Measure extraction quality; enable Grobid if <70% success rate
- **Likely outcome:** ~60% of papers via PMC XML, ~30% via Grobid, ~10% via pymupdf

---

### Issue 5: Knowledge Graph Confidence Scoring (BLOCKING)

**Problem (v1):** Doc says "confidence > 0.7" but never defines how it's calculated.

**Reality:** This is the hardest problem. Naive approaches fail.

#### Bad Approach (Don't Do This):
```python
# WRONG: Assume co-occurrence = relationship
for paper in papers:
    entities_in_paper = extract_entities(paper.text)
    for ent1, ent2 in combinations(entities_in_paper, 2):
        # Create relationship: anything co-occurring gets a relationship
        create_relationship(ent1, ent2, confidence=0.5)
        # Result: KG becomes noise
```

#### Good Approach (Do This):

**Three-Tier Confidence Model:**

```python
# Tier 1: Entity Confidence
entity_confidence = (
    0.7 * ner_score +                    # scispacy confidence
    0.3 * normalization_confidence       # How well we linked to canonical ID
)
# Example: TBX21 extracted with 0.91 NER, normalized to HGNC:11104 with 0.95
#   → entity_confidence = 0.7*0.91 + 0.3*0.95 = 0.922

# Tier 2: Relationship Confidence (LLM-based, explicit extraction)
relationship_confidence = (
    0.5 * llm_extraction_confidence +    # Claude prompt: "extract relationships"
    0.3 * co_occurrence_strength +       # How proximate in text
    0.2 * citation_overlap              # Shared citations increase confidence
)

# Tier 3: Final Score (combined)
final_relationship_score = (
    0.6 * min(entity1_conf, entity2_conf) +  # Both entities must be high confidence
    0.4 * relationship_confidence
)

# Rules (GATES):
if final_relationship_score > 0.80:
    auto_accept_to_kg()                 # High confidence → store immediately
elif 0.50 < final_relationship_score <= 0.80:
    flag_for_human_review()             # Medium confidence → manual validation
    send_to_curation_queue()            # YOU decide
else:
    auto_reject()                        # Low confidence → don't store
```

**LLM-Based Relationship Extraction (Claude):**

```python
async def extract_relationships_llm(paper_text: str, entities: list[Entity]):
    """
    Use Claude to explicitly extract relationships from paper
    """
    prompt = f"""
    You are analyzing a biomedical paper on EBV. 
    
    Entities found in the paper:
    {format_entities(entities)}
    
    Paper abstract:
    {paper_text[:2000]}
    
    Extract relationships between these entities. For each relationship:
    1. State the relationship type (e.g., EXPRESSES, INHIBITS, ASSOCIATES_WITH)
    2. Provide the sentence evidence
    3. Rate confidence (0.5-1.0)
    
    Return as JSON:
    {{
        "relationships": [
            {{
                "entity1": "TBX21",
                "entity2": "B cell",
                "relationship_type": "MARKS_CELL_STATE",
                "evidence": "TBX21 is a marker of atypical B cells...",
                "confidence": 0.92
            }}
        ]
    }}
    """
    
    response = await claude_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return json.loads(response.content[0].text)
```

**Curation Queue (Human-in-the-Loop):**

```python
# PostgreSQL table for pending relationships
CREATE TABLE relationships_pending_review (
    id UUID PRIMARY KEY,
    entity1_id UUID REFERENCES entities(id),
    entity2_id UUID REFERENCES entities(id),
    relationship_type VARCHAR(100),
    confidence_score FLOAT,
    evidence_text TEXT,
    source_paper_doi VARCHAR(1000),
    llm_extraction_json JSONB,
    
    -- Curation fields
    curator_email VARCHAR(255),
    curation_decision VARCHAR(50),  -- 'approved', 'rejected', 'needs_refinement'
    curation_notes TEXT,
    curated_at TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT NOW()
);

# Your curation workflow (weekly)
async def curate_pending_relationships():
    """
    Manually review medium-confidence relationships
    """
    pending = db.query("""
        SELECT * FROM relationships_pending_review
        WHERE curation_decision IS NULL
        AND confidence_score BETWEEN 0.50 AND 0.80
        ORDER BY confidence_score DESC
        LIMIT 20  # Review 20/week
    """)
    
    # For each, you manually approve/reject/refine
    # This data trains a classifier (later automation)
```

**Timeline:** 
- Week 1: Implement LLM extraction + curation queue (5 days)
- Week 2-3: Manually curate 200-300 relationships (10-20/day)
- Week 4: Use curated set to train confidence classifier

**This is honest**: You can't fully automate KG relationships. You curate first, then scale.

---

## Infrastructure & Resource Requirements

### Development Environment (Local)

**Hardware (Your M1 Max with 64GB RAM):**
- RAM: 16GB allocated to containers (healthy margin from 64GB)
- Disk: 150GB for PDFs + local indices + models
- CPU: Multi-core for parallel PDF extraction

**Software Stack:**

```
# Python 3.11+
Python: 3.11.x

# Core dependencies (requirements.txt)
# Data manipulation
numpy, pandas, polars, pyarrow

# Biomedical NLP
spacy, scispacy, sentence-transformers[torch]
allenai/specter2  # Biomedical embeddings

# Databases & search
sqlalchemy, psycopg2-binary          # PostgreSQL ORM
neo4j                                # Graph DB
chromadb                             # Vector store

# PDF & document extraction
pymupdf, pillow, paddleocr          # PDF parsing
# OR: grobid-python-client            # (optional) Grobid integration

# Entity normalization
fuzzy_matcher, thefuzz               # Fuzzy string matching
# + ontology files (HGNC, Cell Ontology, DOID)

# API & Web
fastapi, uvicorn, gunicorn          # REST API
streamlit                            # Web UI
httpx, aiohttp                      # Async HTTP

# Search (updated from v1)
sqlite3                              # Built-in; FTS5 for BM25

# LLM & RAG
langchain, langchain-anthropic      # RAG framework
anthropic                            # Claude API

# Scheduling (separated from API process)
apscheduler, pytz                   # Job scheduling

# Testing & quality
pytest, pytest-asyncio, pytest-cov
black, isort, flake8, mypy
hypothesis                          # Property-based testing

# Monitoring & logging
python-json-logger, prometheus-client

# Portfolio integration (R ↔ Python)
rpy2, reticulate                    # Cross-language calls
```

**Environment Setup:**

```bash
# Reproducible environment via pyproject.toml (poetry) or constraints.txt
python3.11 -m venv venv
source venv/bin/activate
pip install poetry
poetry install  # Locks all dependencies

# OR: Traditional requirements.txt with pin versions
pip install -r requirements.txt
```

---

### Cloud Infrastructure (Free Tier Limits, UPDATED)

#### 1. **Graph Database: Neo4j**

| Feature | Free Tier | Notes |
|---------|-----------|-------|
| **Storage** | 50 GB | Sufficient for 5-10k papers + curated KG |
| **Compute** | Shared (m2-small) | ~100-500ms query latency acceptable |
| **Backup** | Daily automatic | 7-day retention included |
| **Connections** | 1 concurrent | Single-user MVP OK; upgrade when team scales |
| **Cost** | $180/month after free | Plan migration at week 8 |
| **Setup** | 5 minutes | Web console |

**Upgrade Path:**
- **Weeks 0-8:** Neo4j Aura free tier
- **Week 8-10:** Migrate to self-hosted Neo4j Community on VPS (no additional cost)
  ```bash
  # Self-hosted Neo4j on Ubuntu VPS
  apt install openjdk-17-jdk-headless
  wget https://neo4j.com/artifact.php?name=neo4j-community-5.15.0-unix.tar.gz
  tar -xf neo4j-community-5.15.0-unix.tar.gz
  ./neo4j-5.15.0/bin/neo4j console
  ```

---

#### 2. **Vector Database: ChromaDB (Embedded)**

| Feature | ChromaDB Embedded |
|---------|-------------------|
| **Deployment** | In-process library |
| **Storage** | Disk (SQLite backend) |
| **Capacity** | ~500k documents at 768-dim embeddings |
| **Query Latency** | <50ms (single-threaded) |
| **Cost** | FREE (open-source) |
| **Scaling** | Limited to single machine (~5-10 concurrent users) |

**Upgrade Path:**
- Weeks 0-12: Embedded ChromaDB
- Week 12+: Chroma Server (separate container) if hitting concurrency limits

---

#### 3. **Search: SQLite FTS5 (UPDATED from Whoosh)**

| Feature | SQLite FTS5 |
|---------|-------------|
| **Type** | Full-text search (built-in Python) |
| **BM25 Support** | ✅ Yes, configurable |
| **Setup** | No dependencies (built into Python sqlite3) |
| **Indexing Speed** | ~5000 docs/sec (fast) |
| **Query Latency** | <10ms (excellent) |
| **Cost** | FREE |
| **Maintenance** | Actively maintained (SQLite widely used) |

**Why SQLite FTS5 over Whoosh:**
- Whoosh: Dead project (last update 2019)
- SQLite: Maintained by Google, used in Android/iOS
- FTS5: Modern, BM25 ranking built-in
- Integration: One less Python dependency

```python
# app/search/bm25.py
import sqlite3
from fts5_integration import FTS5Manager

# Initialize at startup
db = sqlite3.connect(":memory:" or "/path/to/papers.db")
fts_manager = FTS5Manager(db)

# Index papers
async def index_paper(paper_id: str, text: str, metadata: dict):
    fts_manager.insert(
        rowid=paper_id,
        title=metadata["title"],
        abstract=metadata["abstract"],
        content=text,
        keywords=",".join(metadata.get("keywords", []))
    )

# Query
def search_papers(query: str, limit: int = 10):
    results = db.execute("""
        SELECT rowid, title, abstract, rank
        FROM papers_fts
        WHERE papers_fts MATCH ?
        ORDER BY rank DESC
        LIMIT ?
    """, (query, limit)).fetchall()
    return results
```

---

#### 4. **Object Storage: MinIO (Self-Hosted)**

| Feature | MinIO |
|---------|-------|
| **Type** | S3-compatible object store |
| **Deployment** | Docker container (or standalone binary) |
| **Storage** | Limited by host disk |
| **Cost** | FREE (open-source) |
| **Use Case** | PDF backups, embeddings exports, KG snapshots |

```yaml
# docker-compose.yml addition
services:
  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"  # Web console
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: secure_password
    volumes:
      - minio_data:/minio/data
    command: minio server /minio/data --console-address ":9001"

volumes:
  minio_data:
```

**Backup Strategy:**
- Nightly PostgreSQL dump → MinIO
- Weekly Neo4j export → MinIO
- Monthly full backup → external drive + AWS S3 (as off-site)

---

#### 5. **Compute: VPS (Deployment)**

| Provider | Spec | Cost | Notes |
|----------|------|------|-------|
| **Linode** | 2GB RAM, 1vCPU, 50GB SSD | $12/month | Good API, reliable |
| **DigitalOcean** | 2GB RAM, 1vCPU, 50GB SSD | $6/month | Great docs, $200 credit |
| **Hetzner** | 2GB RAM, 2vCPU, 40GB SSD | €3.99/month (~$4.50) | Best value |
| **Vultr** | 2GB RAM, 1vCPU, 55GB SSD | $6/month | High uptime |

**Deployment Timing:**
- Weeks 0-8: Develop locally
- Week 8-10: Deploy to $6-12/month VPS (production staging)
- Week 12+: Scale if needed (vertical: 4GB → $18-24/month; horizontal: add second instance)

---

#### 6. **LLM API: Claude (Anthropic)**

| Feature | Cost |
|---------|------|
| **Claude API** (Sonnet 4) | Input: $3/MTok, Output: $15/MTok |
| **MVP Usage** | ~100k tokens/day = ~$0.50/day = **$15/month** |
| **Production Usage** | ~500k tokens/day = ~$2.50/day = **$75/month** |
| **Optimization** | Prompt caching (20% savings), batch API (cheaper) |

---

### Infrastructure Cost Summary (UPDATED)

| Layer | Component | MVP Cost | Production Cost | Notes |
|-------|-----------|----------|-----------------|-------|
| **Graph DB** | Neo4j Aura + self-hosted | $0 (free 2 mo) | $0 (self-hosted) | Migrate week 8 |
| **Vector DB** | ChromaDB embedded | $0 | $0 (or $50 if managed) | Scaling optional |
| **Search** | SQLite FTS5 | $0 | $0 | Built-in, free |
| **Storage** | MinIO (self-hosted) | $0 | $0 | Disk only |
| **Compute** | VPS (week 10+) | $0 (local) → $6 | $12-20 | Linode/DigitalOcean |
| **LLM API** | Claude Sonnet 4 | $15 | $75 | Query-dependent |
| **CI/CD** | GitHub Actions | $0 | $0 | Free tier sufficient |
| | | **TOTAL: $15/month** | **$87-95/month** | Both very affordable |

---

## Data Pipeline & ETL (Revised)

### Revised ETL Jobs (With Curation)

#### Job 1: Daily PubMed Crawler

```python
# app/ingestion/pubmed_crawler.py
async def daily_pubmed_ingest():
    """
    Fetch new papers matching EBV query, with validation and dedup
    Trigger: Daily at 02:00 UTC
    """
    
    # Query: EBV papers from last 7 days
    query = """
        ("Epstein-Barr virus"[MeSH] OR EBV[TIAB]) 
        AND (2015:2025[PDAT])
        AND (english[LA])
    """
    
    # Fetch from PubMed
    response = await pubmed_api.search(query, max_results=500)
    
    new_papers = []
    duplicates_found = 0
    validation_errors = 0
    
    for result in response.results:
        paper_dict = {
            "doi": result.get("DOI"),
            "pmid": result.get("PMID"),
            "title": result.get("Title"),
            "abstract": result.get("Abstract", ""),
            "authors": parse_authors(result.get("AuthorList", [])),
            "published_date": parse_date(result.get("PubDate")),
            "journal": result.get("Journal"),
        }
        
        # Validate required fields
        if not all([paper_dict["doi"] or paper_dict["pmid"], paper_dict["title"]]):
            validation_errors += 1
            logger.warning(f"Validation failed: {paper_dict}")
            continue
        
        # Deduplication (by DOI)
        existing = db.papers.find_one({"doi": paper_dict["doi"]})
        if existing:
            duplicates_found += 1
            logger.debug(f"Duplicate found: {paper_dict['doi']}")
            continue
        
        # Try to fetch full-text URL
        pmc_id = get_pmc_id_from_doi(paper_dict["doi"])
        paper_dict["pmc_id"] = pmc_id
        paper_dict["status"] = "new"
        paper_dict["created_at"] = now()
        
        new_papers.append(paper_dict)
    
    # Batch insert
    if new_papers:
        db.papers.insert_many(new_papers)
    
    # Log metrics
    metrics = {
        "papers_fetched": len(response.results),
        "new_papers": len(new_papers),
        "duplicates_found": duplicates_found,
        "validation_errors": validation_errors
    }
    
    db.job_logs.insert_one({
        "job_name": "daily_pubmed_ingest",
        "status": "completed",
        "metrics": metrics,
        "completed_at": now()
    })
    
    # Alert if errors exceed threshold
    if validation_errors > 5:
        alert_slack(f"⚠️ PubMed ingest: {validation_errors} validation errors")
    
    logger.info(f"PubMed ingest completed: {metrics}")
    return metrics
```

---

#### Job 2: PDF Extraction (Hierarchical)

```python
# app/ingestion/pdf_extractor.py
async def batch_pdf_extraction(batch_size: int = 12, timeout: int = 30):
    """
    Extract text from unprocessed papers
    Hierarchy: PMC XML → Grobid → pymupdf + OCR
    """
    
    # Get unprocessed papers
    papers = db.papers.find(
        {"status": "new"},
        limit=batch_size
    )
    
    # Parallel extraction
    async def extract_one(paper):
        try:
            # Step 1: Try PMC XML
            if paper.get("pmc_id"):
                try:
                    xml_content = await fetch_pmc_xml(paper["pmc_id"])
                    text, tables = parse_pmc_xml(xml_content)
                    
                    db.papers.update_one(
                        {"_id": paper["_id"]},
                        {"$set": {
                            "fulltext": text,
                            "tables": tables,
                            "extraction_source": "pmc_xml",
                            "extraction_quality": 0.95,
                            "status": "extracted"
                        }}
                    )
                    return {"status": "success", "source": "pmc_xml"}
                except Exception as e:
                    logger.warning(f"PMC extraction failed for {paper['doi']}: {e}")
            
            # Step 2: Try Grobid
            if paper.get("pdf_url"):
                try:
                    grobid_output = await grobid_client.process_pdf(
                        paper["pdf_url"],
                        timeout=timeout
                    )
                    text, tables, citations = parse_grobid_tei(grobid_output)
                    
                    db.papers.update_one(
                        {"_id": paper["_id"]},
                        {"$set": {
                            "fulltext": text,
                            "tables": tables,
                            "citations": citations,
                            "extraction_source": "grobid",
                            "extraction_quality": 0.85,
                            "status": "extracted"
                        }}
                    )
                    return {"status": "success", "source": "grobid"}
                except Exception as e:
                    logger.warning(f"Grobid extraction failed for {paper['doi']}: {e}")
            
            # Step 3: Fall back to pymupdf + OCR
            try:
                text, tables = await extract_pymupdf_with_ocr(paper["pdf_url"])
                
                db.papers.update_one(
                    {"_id": paper["_id"]},
                    {"$set": {
                        "fulltext": text,
                        "tables": tables,
                        "extraction_source": "pymupdf_ocr",
                        "extraction_quality": 0.60,
                        "status": "extracted"
                    }}
                )
                return {"status": "success", "source": "pymupdf_ocr"}
            except Exception as e:
                logger.error(f"All extraction methods failed for {paper['doi']}: {e}")
                
                db.papers.update_one(
                    {"_id": paper["_id"]},
                    {"$set": {
                        "status": "extraction_failed",
                        "error": str(e)
                    }}
                )
                return {"status": "failed", "error": str(e)}
        
        except Exception as e:
            logger.error(f"Extraction worker error: {e}")
            return {"status": "worker_error", "error": str(e)}
    
    # Run in parallel (12 concurrent)
    results = await asyncio.gather(
        *[extract_one(paper) for paper in papers],
        return_exceptions=True
    )
    
    # Log results
    success_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
    failure_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") != "success")
    
    db.job_logs.insert_one({
        "job_name": "batch_pdf_extraction",
        "status": "completed",
        "records_processed": len(papers),
        "success": success_count,
        "failures": failure_count,
        "completed_at": now()
    })
    
    return {"success": success_count, "failed": failure_count}
```

---

#### Job 3: Entity Extraction + Normalization

```python
# app/ingestion/entity_extractor.py
async def extract_and_normalize_entities(batch_size: int = 50):
    """
    Extract entities from extracted papers, normalize to canonical IDs
    """
    
    papers = db.papers.find(
        {"status": "extracted", "entities_extracted": {"$ne": True}},
        limit=batch_size
    )
    
    for paper in papers:
        text = paper["fulltext"]
        
        # Step 1: NER via scispacy
        entities_raw = ner_model(text)
        # Output: [(text, label, confidence), ...]
        
        # Step 2: Normalize via EntityNormalizer
        normalizer = EntityNormalizer(ontologies=load_ontologies())
        canonical_entities = []
        
        for entity_text, entity_type, ner_conf in entities_raw:
            normalized = normalizer.resolve(entity_text, entity_type)
            
            if normalized.get("confidence", 0) < 0.3:
                # Low confidence → flag for review
                db.entities_pending_review.insert_one({
                    "paper_doi": paper["doi"],
                    "raw_text": entity_text,
                    "entity_type": entity_type,
                    "ner_confidence": ner_conf,
                    "normalization_confidence": normalized.get("confidence", 0),
                    "canonical_id": normalized.get("canonical_id"),
                    "status": "pending_human_review"
                })
                continue
            
            canonical_entities.append({
                "paper_id": paper["_id"],
                "canonical_id": normalized["canonical_id"],
                "symbol": normalized["symbol"],
                "entity_type": entity_type,
                "aliases": normalized.get("aliases", []),
                "db_xrefs": normalized.get("db_xref", {}),
                "confidence": normalized["confidence"],
                "source": normalized.get("source")
            })
        
        # Step 3: Upsert to PostgreSQL (using canonical IDs)
        for entity in canonical_entities:
            db.entities.upsert(
                on_conflict="canonical_id",
                values=entity
            )
        
        # Mark paper as processed
        db.papers.update_one(
            {"_id": paper["_id"]},
            {"$set": {"entities_extracted": True}}
        )
    
    logger.info(f"Entity extraction complete: {len(papers)} papers processed")
```

---

#### Job 4: Relationship Extraction (LLM + Curation)

```python
# app/ingestion/relationship_extractor.py
async def extract_relationships_llm(batch_size: int = 20):
    """
    Use Claude to extract relationships from papers
    Flag all results for curation (don't auto-add to KG)
    """
    
    papers = db.papers.find(
        {"entities_extracted": True, "relationships_extracted": {"$ne": True}},
        limit=batch_size
    )
    
    for paper in papers:
        # Get entities from this paper
        paper_entities = db.entities.find({"paper_id": paper["_id"]})
        
        if not paper_entities:
            continue  # Skip if no entities
        
        # Create prompt
        prompt = f"""
        You are analyzing a biomedical paper on EBV research.
        
        PAPER TITLE: {paper["title"]}
        ABSTRACT: {paper["abstract"][:1500]}
        
        ENTITIES FOUND:
        {format_entities(paper_entities)}
        
        Extract explicit biological relationships between these entities. 
        For each relationship, provide:
        1. Two entity names
        2. Relationship type (EXPRESSES, REGULATES, INHIBITS, ASSOCIATES_WITH, DERIVES_FROM, etc.)
        3. Evidence sentence from the paper
        4. Your confidence (0.5-1.0)
        
        Return ONLY valid JSON:
        {{
            "relationships": [
                {{
                    "entity1": "TBX21",
                    "entity2": "ABC",
                    "relationship_type": "MARKS_CELL_STATE",
                    "evidence": "...",
                    "confidence": 0.92
                }}
            ]
        }}
        """
        
        try:
            response = await claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            relationships_json = json.loads(response.content[0].text)
            
            # Store ALL relationships in pending review (don't auto-add to KG yet)
            for rel in relationships_json["relationships"]:
                db.relationships_pending_review.insert_one({
                    "paper_doi": paper["doi"],
                    "entity1": rel["entity1"],
                    "entity2": rel["entity2"],
                    "relationship_type": rel["relationship_type"],
                    "evidence": rel["evidence"],
                    "llm_confidence": rel["confidence"],
                    "combined_confidence": rel["confidence"],  # Will update after curation
                    "status": "pending_curation",
                    "created_at": now()
                })
        
        except Exception as e:
            logger.error(f"LLM extraction failed for {paper['doi']}: {e}")
        
        # Mark paper as processed
        db.papers.update_one(
            {"_id": paper["_id"]},
            {"$set": {"relationships_extracted": True}}
        )
    
    logger.info(f"LLM extraction complete: relationships awaiting curation")
```

---

#### Job 5: Embedding Generation

```python
# app/ingestion/embeddings_pipeline.py
async def generate_embeddings(model: str = "allenai/specter2", batch_size: int = 100):
    """
    Embed paper chunks using biomedical sentence transformer
    """
    
    # Load model (cached after first load)
    embedding_model = SentenceTransformer(model)
    
    # Get unembedded chunks
    chunks = db.paper_chunks.find(
        {"embedding_status": "pending"},
        limit=batch_size
    )
    
    chunk_texts = [chunk["text"] for chunk in chunks]
    
    try:
        # Batch encode
        embeddings = embedding_model.encode(
            chunk_texts,
            batch_size=32,  # Process 32 at a time
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Store in ChromaDB
        chromadb_client.add(
            ids=[chunk["_id"] for chunk in chunks],
            embeddings=embeddings.tolist(),
            metadatas=[{
                "paper_doi": chunk["paper_doi"],
                "chunk_index": chunk["chunk_index"],
                "source_type": chunk["source_type"]
            } for chunk in chunks]
        )
        
        # Mark as embedded in PostgreSQL
        for chunk in chunks:
            db.paper_chunks.update_one(
                {"_id": chunk["_id"]},
                {"$set": {"embedding_status": "embedded"}}
            )
        
        logger.info(f"Embedded {len(chunks)} chunks")
    
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise
```

---

#### Job 6: Materialization (PostgreSQL → Neo4j/ChromaDB)

```python
# app/materialization/graph_materializer.py
async def materialize_neo4j_from_postgres():
    """
    Nightly job: Rebuild Neo4j from PostgreSQL source of truth
    Ensures consistency, allows rollback
    """
    
    logger.info("Starting Neo4j materialization...")
    
    # Clear existing graph (or use versioning)
    neo4j_client.clear_graph()
    
    # Load curated entities from PostgreSQL
    entities = db.entities.find({"confidence": {"$gte": 0.7}})
    
    for entity in entities:
        neo4j_client.query("""
            MERGE (e:Entity {canonical_id: $cid})
            SET e.symbol = $symbol,
                e.entity_type = $type,
                e.aliases = $aliases,
                e.db_xrefs = $xrefs,
                e.confidence = $conf,
                e.updated_at = timestamp()
        """, {
            "cid": entity["canonical_id"],
            "symbol": entity["symbol"],
            "type": entity["entity_type"],
            "aliases": entity.get("aliases", []),
            "xrefs": entity.get("db_xrefs", {}),
            "conf": entity["confidence"]
        })
    
    # Load curated relationships from PostgreSQL
    relationships = db.relationships_pending_review.find({
        "status": "approved_for_kg",
        "combined_confidence": {"$gte": 0.70}
    })
    
    for rel in relationships:
        neo4j_client.query("""
            MATCH (e1:Entity {symbol: $e1}), (e2:Entity {symbol: $e2})
            MERGE (e1) -[r:RELATIONSHIP]-> (e2)
            SET r.type = $type,
                r.confidence = $conf,
                r.evidence = $evidence,
                r.paper_doi = $doi,
                r.updated_at = timestamp()
        """, {
            "e1": rel["entity1"],
            "e2": rel["entity2"],
            "type": rel["relationship_type"],
            "conf": rel["combined_confidence"],
            "evidence": rel["evidence"],
            "doi": rel["paper_doi"]
        })
    
    # Load papers
    papers = db.papers.find({"status": {"$in": ["extracted", "processed"]}})
    
    for paper in papers:
        neo4j_client.query("""
            MERGE (p:Paper {doi: $doi})
            SET p.title = $title,
                p.pmid = $pmid,
                p.year = year($date),
                p.journal = $journal
        """, {
            "doi": paper["doi"],
            "title": paper["title"],
            "pmid": paper.get("pmid"),
            "date": paper["published_date"],
            "journal": paper.get("journal")
        })
        
        # Link paper to entities
        for entity in db.entities.find({"paper_id": paper["_id"]}):
            neo4j_client.query("""
                MATCH (p:Paper {doi: $doi}), (e:Entity {canonical_id: $cid})
                MERGE (p) -[d:DESCRIBES]-> (e)
            """, {"doi": paper["doi"], "cid": entity["canonical_id"]})
    
    logger.info("Neo4j materialization complete")
    
    # Log success
    db.job_logs.insert_one({
        "job_name": "materialize_neo4j",
        "status": "completed",
        "completed_at": now()
    })
```

---

## Knowledge Graph Design & Confidence Scoring

### KG Schema (Entity Types & Relationships)

**Entity Types:**

```python
ENTITY_TYPES = {
    "GENE": {
        "examples": ["EBNA1", "IRF4", "TBX21"],
        "ontology": "HGNC",
        "properties": ["symbol", "ensembl_id", "aliases", "ncbi_gene_id"]
    },
    "PROTEIN": {
        "examples": ["EBNA-1", "TIM-3", "PD-1"],
        "ontology": "UniProt",
        "properties": ["uniprot_id", "aliases", "molecular_weight"]
    },
    "CELL_TYPE": {
        "examples": ["B cell", "TBX21+ ABC", "Plasma cell"],
        "ontology": "Cell Ontology (CL)",
        "properties": ["cl_id", "canonical_name", "markers"]
    },
    "DISEASE": {
        "examples": ["EBV infection", "Lymphoma", "MS"],
        "ontology": "DOID",
        "properties": ["doid", "mesh_terms", "icd10_codes"]
    },
    "TISSUE": {
        "examples": ["Lymph node", "Bone marrow", "Spleen"],
        "ontology": "UBERON",
        "properties": ["uberon_id", "synonyms"]
    },
    "PATHWAY": {
        "examples": ["BCR signaling", "NF-κB pathway"],
        "ontology": "Reactome, KEGG",
        "properties": ["reactome_id", "kegg_id", "genes"]
    },
    "DATASET": {
        "examples": ["GSE189141", "SRR123456"],
        "ontology": "GEO, SRA",
        "properties": ["accession", "technique", "organism", "sample_count"]
    },
    "PAPER": {
        "examples": ["10.1234/example", "PMID:12345"],
        "properties": ["doi", "pmid", "title", "year", "journal"]
    },
    "AUTHOR": {
        "examples": ["John Smith"],
        "properties": ["name", "orcid", "institution"]
    }
}
```

**Relationship Types (Curated List, High-Confidence Only):**

```
Biological Relationships:
- GENE_ENCODES_PROTEIN       # IL2RG encodes IL-2 receptor gamma
- PROTEIN_MARKS_CELL_TYPE    # TIM-3 marks exhausted T cells
- GENE_EXPRESSES_IN           # TBX21 expresses in B cells
- PROTEIN_INHIBITS_PROTEIN    # PD-1 inhibits T cell activation
- PATHWAY_CONTAINS_GENE       # NF-κB pathway contains RELA
- CELL_TYPE_DERIVES_FROM      # Plasma cell derives from B cell

Disease Relationships:
- DISEASE_ASSOCIATED_WITH_GENE  # Lymphoma associated with TP53
- DISEASE_AFFECTS_TISSUE        # EBV infects lymphoid tissues
- DISEASE_CHARACTERIZED_BY      # ABC characterized by TBX21+ phenotype

Epidemiological:
- PAPER_STUDIES_DISEASE       # This paper studies EBV infection
- DATASET_MEASURES_CELL_TYPE  # GSE189141 measures B cell subtypes
- AUTHOR_PUBLISHED            # John Smith published this paper

Negation (Explicit):
- GENE_NOT_EXPRESSED_IN       # TBX21 NOT expressed in naive B cells
- PROTEIN_DOES_NOT_INHIBIT    # Drug X does NOT inhibit PD-1
```

**Confidence Scoring Function (Final):**

```python
def calculate_relationship_confidence(
    entity1_conf: float,
    entity2_conf: float,
    llm_extraction_conf: float,
    co_occurrence_strength: float,  # 0-1 based on sentence proximity
    citation_overlap: float,        # 0-1 based on shared citations
) -> float:
    """
    Calculate final relationship confidence
    """
    
    # Ensure both entities are well-established
    entity_min_conf = min(entity1_conf, entity2_conf)
    if entity_min_conf < 0.60:
        return 0.0  # Too weak; don't create relationship
    
    # Weight components
    relationship_conf = (
        0.5 * llm_extraction_conf +      # LLM explicitly said this relationship
        0.3 * co_occurrence_strength +   # How close in text
        0.2 * citation_overlap           # How many shared citations
    )
    
    # Penalize if entities are weak (even if relationship is strong)
    final_conf = relationship_conf * (0.7 + 0.3 * entity_min_conf)
    
    return min(final_conf, 1.0)  # Clamp to [0, 1]

# Decision gates
def relationship_decision(confidence: float) -> str:
    if confidence > 0.80:
        return "auto_accept"           # High confidence → add to KG
    elif 0.50 <= confidence <= 0.80:
        return "human_review"          # Medium → you decide
    else:
        return "auto_reject"           # Low → don't add
```

---

## Entity Normalization & Curation

### Curation Dashboard (Streamlit UI)

```python
# app/ui/curation_dashboard.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="EBV RAG Curation", layout="wide")

# Authentication
user_email = st.session_state.get("email", None)
if not user_email:
    st.stop()

# Sidebar: Queue selection
queue_type = st.sidebar.radio(
    "Select Curation Queue",
    ["Entities Pending Review", "Relationships Pending Review", "Statistics"]
)

if queue_type == "Entities Pending Review":
    st.header("Unresolved Entity Extraction")
    
    # Fetch pending entities
    pending = db.entities_pending_review.find(
        {"status": "pending_human_review"}
    ).limit(20)
    
    pending_df = pd.DataFrame(pending)
    st.dataframe(pending_df[["raw_text", "entity_type", "ner_confidence", "canonical_id"]])
    
    # For each pending entity, allow user to:
    # 1. Approve auto-normalization
    # 2. Manually select correct canonical ID
    # 3. Mark as "unknown" (create new unresolved entity)
    
    for idx, row in pending_df.iterrows():
        with st.expander(f"{row['raw_text']} ({row['entity_type']})"):
            st.write(f"**NER Confidence**: {row['ner_confidence']:.2f}")
            st.write(f"**Paper**: {row['paper_doi']}")
            
            # Show alternatives from ontology
            if row["entity_type"] == "GENE":
                ontology = load_hgnc()
                matches = ontology.fuzzy_search(row["raw_text"], threshold=0.7)
                
                choice = st.selectbox(
                    "Select canonical ID",
                    options=[m["ensembl_id"] for m in matches],
                    format_func=lambda x: next(m["symbol"] for m in matches if m["ensembl_id"] == x)
                )
                
                if st.button(f"Approve: {choice}"):
                    db.entities_pending_review.update_one(
                        {"_id": row["_id"]},
                        {"$set": {
                            "status": "approved",
                            "canonical_id": choice,
                            "curator_email": user_email,
                            "curated_at": now()
                        }}
                    )
                    st.success("Entity approved!")
            
            if st.button(f"Mark as Unresolvable"):
                db.entities_pending_review.update_one(
                    {"_id": row["_id"]},
                    {"$set": {
                        "status": "unresolvable",
                        "curator_email": user_email
                    }}
                )
                st.warning("Entity marked as unresolvable")

elif queue_type == "Relationships Pending Review":
    st.header("Relationships Awaiting Curation")
    
    # Fetch pending relationships
    pending_rels = db.relationships_pending_review.find(
        {"status": "pending_curation"}
    ).limit(20)
    
    for rel in pending_rels:
        with st.expander(f"{rel['entity1']} → {rel['entity2']}"):
            st.write(f"**Relationship Type**: {rel['relationship_type']}")
            st.write(f"**LLM Confidence**: {rel['llm_confidence']:.2f}")
            st.write(f"**Evidence**: {rel['evidence']}")
            st.write(f"**Paper**: {rel['paper_doi']}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button(f"✅ Approve", key=f"approve_{rel['_id']}"):
                    db.relationships_pending_review.update_one(
                        {"_id": rel["_id"]},
                        {"$set": {
                            "status": "approved_for_kg",
                            "combined_confidence": rel["llm_confidence"],
                            "curator_email": user_email,
                            "curated_at": now()
                        }}
                    )
                    st.success("Relationship approved!")
            
            with col2:
                if st.button(f"❌ Reject", key=f"reject_{rel['_id']}"):
                    db.relationships_pending_review.update_one(
                        {"_id": rel["_id"]},
                        {"$set": {
                            "status": "rejected",
                            "curator_email": user_email
                        }}
                    )
                    st.error("Relationship rejected")
            
            with col3:
                if st.button(f"⚠️ Needs Refinement", key=f"refine_{rel['_id']}"):
                    new_conf = st.slider("Adjust confidence", 0.0, 1.0, rel["llm_confidence"])
                    db.relationships_pending_review.update_one(
                        {"_id": rel["_id"]},
                        {"$set": {
                            "status": "needs_refinement",
                            "combined_confidence": new_conf,
                            "curator_email": user_email
                        }}
                    )
                    st.info("Confidence adjusted")

elif queue_type == "Statistics":
    st.header("Curation Statistics")
    
    stats = {
        "Total Papers": db.papers.count_documents({}),
        "Extracted Papers": db.papers.count_documents({"status": "extracted"}),
        "Entities Extracted": db.entities.count_documents({}),
        "Entities Pending Review": db.entities_pending_review.count_documents({"status": "pending_human_review"}),
        "Relationships Pending": db.relationships_pending_review.count_documents({"status": "pending_curation"}),
        "Relationships Approved": db.relationships_pending_review.count_documents({"status": "approved_for_kg"}),
    }
    
    st.metric("Curation Progress", f"{stats['Relationships Approved']}/{stats['Relationships Pending']}")
    
    # Chart: Approval rate over time
    approvals_over_time = db.relationships_pending_review.aggregate([
        {"$match": {"curated_at": {"$exists": True}}},
        {"$group": {
            "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$curated_at"}},
            "approved": {"$sum": {"$cond": [{"$eq": ["$status", "approved_for_kg"]}, 1, 0]}},
            "rejected": {"$sum": {"$cond": [{"$eq": ["$status", "rejected"]}, 1, 0]}}
        }},
        {"$sort": {"_id": 1}}
    ])
    
    st.line_chart(list(approvals_over_time))
```

---

## API & Query Layer Design

### REST API (FastAPI) — Updated

```python
# app/api/routes.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse

app = FastAPI(title="EBV RAG API", version="2.0")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    search_type: str = "hybrid"  # 'keyword', 'semantic', 'graph', 'hybrid'
    include_citations: bool = True

class RagResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    retrieved_documents: list[dict]
    citations: list[dict]
    generation_time_s: float

@app.post("/query/hybrid")
async def hybrid_query(req: QueryRequest) -> RagResponse:
    """
    Hybrid search with confidence-filtered KG
    """
    start = time.time()
    
    # 1. Retrieve from all three modalities (parallel)
    bm25_results = await retrieve_bm25(req.query, k=req.top_k*2)
    vector_results = await retrieve_vector(req.query, k=req.top_k*2)
    graph_results = await retrieve_graph(req.query, k=req.top_k*2)
    
    # 2. Merge & rank
    merged = merge_results(bm25_results, vector_results, graph_results)
    top_k = rank_and_slice(merged, k=req.top_k)
    
    # 3. Synthesize with Claude
    answer = await synthesize_with_claude(req.query, top_k)
    
    # 4. Extract citations
    citations = extract_citations(answer.content, top_k)
    
    elapsed = time.time() - start
    
    return RagResponse(
        query=req.query,
        answer=answer.content,
        confidence=answer.confidence,
        retrieved_documents=[{"title": d["title"], "doi": d["doi"], "score": d["score"]} for d in top_k],
        citations=citations,
        generation_time_s=elapsed
    )

@app.get("/graph/explore/{entity_id}")
async def explore_entity(entity_id: str, hops: int = 2):
    """
    Explore KG around entity, returning only HIGH-CONFIDENCE edges
    """
    subgraph = neo4j_client.query(f"""
        MATCH (e:Entity {{symbol: $eid}})
        CALL apoc.path.subgraphAll(e, {{relationshipFilter: "RELATIONSHIP", maxLevel: {hops}}})
        YIELD nodes, relationships
        RETURN nodes, [r IN relationships WHERE r.confidence > 0.70] as high_conf_rels
    """, {"eid": entity_id})
    
    return {
        "entity": entity_id,
        "subgraph": subgraph
    }

@app.get("/portfolio/markers")
async def get_portfolio_markers(disease: str = None, cell_type: str = None):
    """
    Query portfolio project marker genes
    """
    query = {}
    if disease:
        query["disease"] = disease
    if cell_type:
        query["cell_type"] = cell_type
    
    markers = db.portfolio_markers.find(query)
    return list(markers)

@app.get("/admin/curation-status")
async def curation_status():
    """System curation status"""
    return {
        "entities_pending": db.entities_pending_review.count_documents({"status": "pending_human_review"}),
        "relationships_pending": db.relationships_pending_review.count_documents({"status": "pending_curation"}),
        "relationships_approved": db.relationships_pending_review.count_documents({"status": "approved_for_kg"}),
        "kgnodes": neo4j_client.query("MATCH (n:Entity) RETURN count(n) as count")[0]["count"],
        "kgedges": neo4j_client.query("MATCH ()-[r:RELATIONSHIP]->() WHERE r.confidence > 0.70 RETURN count(r) as count")[0]["count"],
    }
```

---

## Deployment & DevOps

### Docker Compose (Updated with Scheduler Separation)

```yaml
# docker-compose.yml
version: '3.8'

services:
  # API Server
  api:
    build: .
    command: gunicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/ebv_rag
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - GROBID_URL=http://grobid:8070  # Optional
    depends_on:
      - postgres
      - neo4j
    volumes:
      - ./data/chromadb:/app/data/chromadb
      - ./data/whoosh_index:/app/data/whoosh_index
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Background Scheduler (SEPARATE from API)
  scheduler:
    build: .
    command: python app/scheduler/run.py
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/ebv_rag
      - NEO4J_URI=bolt://neo4j:7687
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    depends_on:
      - postgres
      - neo4j
    restart: always

  # Streamlit UI
  ui:
    build: .
    command: streamlit run app/ui/curation_dashboard.py --server.port 8501
    ports:
      - "8501:8501"
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/ebv_rag
    depends_on:
      - postgres

  # PostgreSQL (Source of Truth)
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: ebv_rag
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  # Neo4j (Materialized View)
  neo4j:
    image: neo4j:5.15
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}
      NEO4J_PLUGINS: '["apoc"]'
    volumes:
      - neo4j_data:/var/lib/neo4j/data
    ports:
      - "7687:7687"
      - "7474:7474"

  # MinIO (S3-compatible backup storage)
  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: ${MINIO_PASSWORD}
    volumes:
      - minio_data:/minio/data
    command: minio server /minio/data --console-address ":9001"

  # Grobid (Optional, for advanced PDF extraction)
  grobid:
    image: grobid/grobid:0.8.0
    ports:
      - "8070:8070"
    environment:
      - JAVA_OPTS=-Xmx4g
    profiles: ["optional"]  # Enable with: docker-compose --profile optional up

volumes:
  postgres_data:
  neo4j_data:
  minio_data:
```

---

### GitHub Actions CI/CD (Updated)

```yaml
# .github/workflows/tests.yml
name: Tests & Quality Checks

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_ebv_rag
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-retries 5
        ports:
          - 5432:5432
      
      neo4j:
        image: neo4j:5.15
        env:
          NEO4J_AUTH: neo4j/testpass
        ports:
          - 7687:7687
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Cache pip packages
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio pytest-cov
      
      - name: Run linting
        run: |
          flake8 app/ --max-line-length=120
          black --check app/
          mypy app/ --ignore-missing-imports
      
      - name: Run tests
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_ebv_rag
          NEO4J_URI: bolt://localhost:7687
          NEO4J_USER: neo4j
          NEO4J_PASSWORD: testpass
        run: |
          pytest tests/ --cov=app --cov-report=xml -v
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          flags: unittests
          fail_ci_if_error: false
```

---

## Testing & Evaluation

### Benchmark Dataset (Gold Standard)

Create a manually curated evaluation set:

```json
{
  "queries": [
    {
      "id": "q001",
      "query": "What is the role of TBX21 in EBV-driven B cell transformation?",
      "expected_papers": [
        "doi:10.1234/xxx",
        "doi:10.1234/yyy"
      ],
      "expected_answer_keywords": ["TBX21", "atypical B cells", "transcription factor"],
      "category": "biology"
    },
    {
      "id": "q002",
      "query": "Which datasets study EBV+ DLBCL spatial transcriptomics?",
      "expected_datasets": ["GSE274051"],
      "expected_papers": ["doi:10.1234/zzz"],
      "category": "data_discovery"
    }
  ]
}
```

### Evaluation Metrics

```python
# app/evaluation/metrics.py

def retrieval_precision_at_k(expected_docs: set, retrieved_docs: list, k: int = 5) -> float:
    """What % of top-k are relevant?"""
    top_k_ids = {d["id"] for d in retrieved_docs[:k]}
    relevant = len(top_k_ids & expected_docs)
    return relevant / k

def retrieval_recall_at_k(expected_docs: set, retrieved_docs: list, k: int = 10) -> float:
    """What % of all relevant docs appear in top-k?"""
    top_k_ids = {d["id"] for d in retrieved_docs[:k]}
    relevant = len(top_k_ids & expected_docs)
    return relevant / len(expected_docs) if expected_docs else 0

def ndcg_score(expected_docs: dict[str, float], retrieved_docs: list, k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain (with relevance scores)"""
    dcg = sum(
        (expected_docs.get(doc["id"], 0) / (i + 1))
        for i, doc in enumerate(retrieved_docs[:k])
    )
    ideal_dcg = sum(
        (rel / (i + 1))
        for i, rel in enumerate(sorted(expected_docs.values(), reverse=True)[:k])
    )
    return dcg / ideal_dcg if ideal_dcg > 0 else 0

def citation_accuracy(answer_text: str, citations: list, retrieved_docs: dict) -> float:
    """% of cited statements that actually appear in retrieved docs"""
    correct = 0
    for citation in citations:
        doc = retrieved_docs.get(citation["doi"])
        if doc and citation["text"] in doc["content"]:
            correct += 1
    return correct / len(citations) if citations else 0

def answer_quality_llm(query: str, generated_answer: str, reference_answer: str) -> dict:
    """Use Claude to evaluate answer quality"""
    eval_prompt = f"""
    Query: {query}
    Generated Answer: {generated_answer}
    Reference Answer: {reference_answer}
    
    Rate on 0-1:
    - Relevance (answers the query)
    - Accuracy (no hallucinations)
    - Completeness (covers key points)
    - Clarity (well-written)
    
    Return JSON: {{"relevance": X, "accuracy": Y, "completeness": Z, "clarity": W}}
    """
    
    response = claude_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=200,
        messages=[{"role": "user", "content": eval_prompt}]
    )
    
    return json.loads(response.content[0].text)

async def eval_on_benchmark(benchmark_path: str):
    """Run full evaluation on benchmark set"""
    with open(benchmark_path) as f:
        benchmark = json.load(f)
    
    results = []
    
    for query_item in benchmark["queries"]:
        response = await hybrid_query(QueryRequest(query=query_item["query"]))
        
        retrieved_ids = {d["doi"] for d in response.retrieved_documents}
        expected_ids = set(query_item.get("expected_papers", []))
        
        metrics = {
            "query_id": query_item["id"],
            "precision@5": retrieval_precision_at_k(expected_ids, response.retrieved_documents, k=5),
            "recall@10": retrieval_recall_at_k(expected_ids, response.retrieved_documents, k=10),
            "ndcg@10": ndcg_score({d: 1.0 for d in expected_ids}, response.retrieved_documents),
            "citation_accuracy": citation_accuracy(response.answer, response.citations, {d["doi"]: d for d in response.retrieved_documents}),
        }
        
        results.append(metrics)
    
    # Summary
    summary = {
        "avg_precision@5": np.mean([r["precision@5"] for r in results]),
        "avg_recall@10": np.mean([r["recall@10"] for r in results]),
        "avg_ndcg@10": np.mean([r["ndcg@10"] for r in results]),
        "avg_citation_accuracy": np.mean([r["citation_accuracy"] for r in results]),
    }
    
    return {"results": results, "summary": summary}
```

---

## Cost Analysis (UPDATED)

### MVP (12 weeks)

| Component | Cost | Notes |
|-----------|------|-------|
| **Local Dev** | $0 | Your M1 Max |
| **Neo4j Aura** | $0 → $180 | Free tier for 2 months; migrate week 8 |
| **VPS (Linode)** | $0 → $72 | Deploy week 10; $6/month |
| **Claude API** | $15-20 | ~100k tokens/day |
| **Domain (optional)** | $0 | Skip MVP |
| **Grobid (optional)** | $0 | Docker, self-hosted |
| | **Total: $15-20/month** | Very affordable |

### Production (6+ months)

| Component | Cost | Notes |
|-----------|------|-------|
| **VPS (4GB)** | $18-24 | Vultr, DigitalOcean |
| **PostgreSQL (managed)** | $50 | DigitalOcean Spaces |
| **Neo4j (self-hosted)** | $0 | VPS disk only |
| **Claude API** | $75-150 | ~500k tokens/day |
| **Monitoring** | $0 | Prometheus + Grafana (self-hosted) |
| | **Total: $143-224/month** | Still very reasonable |

---

## Known Gaps & Phase 1 Validation Tasks

**These MUST be addressed before full implementation:**

### Critical Blocking Issues

- [ ] **KG Confidence Scoring**: Validate confidence formula on 200+ curated relationships
- [ ] **Entity Normalization**: Complete integration of HGNC, Cell Ontology, DOID; test fuzzy matching
- [ ] **Database State Consistency**: Test PostgreSQL ↔ Neo4j ↔ ChromaDB sync; design rollback procedures
- [ ] **Scheduler Separation**: Verify APScheduler in separate container doesn't cause duplicate executions
- [ ] **Embedding Model Validation**: Compare specter2 vs. all-MiniLM on EBV-specific queries; measure semantic recall

### Important (Pre-Production)

- [ ] **PDF Extraction Hierarchy**: Test PMC XML → Grobid → pymupdf fallback chain on 100 papers; measure success rate
- [ ] **Curation Dashboard UX**: Get feedback from users on entity/relationship review workflow
- [ ] **Portfolio Integration**: Design AnnData import pipeline; validate cell type + marker gene ingestion
- [ ] **Citation Extraction Accuracy**: Improve Claude prompt for citation identification; target >90% accuracy
- [ ] **Performance Profiling**: Measure query latency at 50k documents; optimize if >10s

### Nice-to-Have (Post-MVP)

- [ ] **Elasticsearch Integration**: Migrate from SQLite FTS5 if searching >1M documents
- [ ] **Kubernetes Deployment**: Only if needing auto-scaling (unlikely for this domain size)
- [ ] **LLM Fine-Tuning**: Train specialized model on EBV domain (low priority)

---

## Version Control & Changelog

### Version History

**v1.0 (2025-05-04, Initial Draft)**
- High-level architecture overview
- Infrastructure choices (Neo4j, ChromaDB, Whoosh, VPS)
- Basic ETL pipeline
- Naive assumption: fully automated KG building

**v2.0 (2025-05-04, Critique-Integrated)**
- **Added**: Critical biology & data science sections addressing:
  - Seurat/R to Python integration (mandate AnnData/CSV exports)
  - Entity normalization & synonymy management (Bern2, ontology integration)
  - Embedding model choice (upgraded to specter2)
  - PDF extraction hierarchy (PMC XML → Grobid → pymupdf)
  - KG confidence scoring (three-tier model, human curation gates)
  - Database state consistency (PostgreSQL as source of truth)
  - Scheduler separation (prevent duplicate executions)
  
- **Removed**: Whoosh search (replaced with SQLite FTS5)
- **Modified**:
  - ETL pipeline now includes human curation queue
  - Neo4j/ChromaDB are materialized views (not primary stores)
  - Relationship extraction uses Claude LLM explicitly
  - Curation dashboard added for manual approval workflow
  
- **Cost**: Minimal change (~$15-20/month MVP, $143-224/month production)
- **Timeline**: Added 1-2 weeks for curation infrastructure
- **Key Insight**: MVP is NOT about fully automated KG; it's about infrastructure to manage a curated, growing KG

### Feedback Integration

Critique Source: Senior engineer with biology + data science background

Key Valid Points Addressed:
1. ✅ Removed naive "auto-extract" assumption; added human-in-the-loop
2. ✅ Integrated entity normalization with standard ontologies
3. ✅ Upgraded embedding model to biomedical-tuned (specter2)
4. ✅ Added hierarchical PDF extraction (PMC → Grobid → fallback)
5. ✅ Defined explicit confidence scoring with decision gates
6. ✅ Fixed database state sync (PostgreSQL as single source of truth)
7. ✅ Separated scheduler from API process
8. ✅ Replaced Whoosh with maintained SQLite FTS5
9. ✅ Clarified Seurat/R integration (mandate data export formats)

### Next Iteration (v2.1 Targets)

Likely topics for next refinement:
- [ ] **Neo4j Materialization**: Detailed algorithm for nightly rebuild without data loss
- [ ] **Relationship Extraction**: Specific Claude prompts; validate on 50 papers
- [ ] **Curation Workflows**: User testing of dashboard; feedback loops
- [ ] **Performance Bottlenecks**: Query latency targets; indexing strategy
- [ ] **Portfolio Integration**: Concrete AnnData ↔ PostgreSQL mapping
- [ ] **Data Quality Pipeline**: Comprehensive validation checks; alerting thresholds

---

## Appendix A: Reference Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                     USER INTERFACES (Week 12+)                        │
├────────────────────────────────────────────────────────────────────────┤
│   REST API (FastAPI)  │   Streamlit UI   │   Jupyter Notebooks       │
│   /query/hybrid       │   Curation Dash  │   Research Scripts        │
└──────────────┬────────────────┬──────────────────┬────────────────────┘
               │                │                  │
               └────────────────┼──────────────────┘
                                │
        ┌───────────────────────▼────────────────────────┐
        │   QUERY & SYNTHESIS LAYER (Week 9-10)         │
        ├────────────────────────────────────────────────┤
        │ Hybrid Search Orchestrator                     │
        │ • SQLite FTS5 (BM25)                          │
        │ • ChromaDB (semantic, specter2)               │
        │ • Neo4j Cypher (graph traversal)              │
        │ • Ranking & merging                           │
        │                                                │
        │ LLM Chain (Claude Sonnet 4)                    │
        │ • Retrieval ranking                            │
        │ • Multi-hop reasoning                          │
        │ • Citation extraction                          │
        │ • Confidence estimation                        │
        └────────────┬──────────────────────────────────┘
                     │
    ┌────────────────▼─────────────────────────────┐
    │  KNOWLEDGE REPRESENTATION (Materialized)     │
    ├──────────────────────────────────────────────┤
    │ PostgreSQL (Week 5: Source of Truth)         │
    │ • Papers, chunks, entities                   │
    │ • Relationships (approved + pending)         │
    │ • Audit trail, curation logs                 │
    │                                              │
    │ Neo4j (Week 7: Materialized View)            │
    │ • Entity nodes (curated)                     │
    │ • High-confidence relationships only         │
    │ • Rebuilt nightly from PostgreSQL            │
    │                                              │
    │ ChromaDB (Week 8: Materialized View)         │
    │ • specter2 embeddings                        │
    │ • Document chunks + metadata                 │
    │ • Rebuilt with new extractions               │
    │                                              │
    │ SQLite (Week 6: Materialized View)           │
    │ • FTS5 indexed papers                        │
    │ • BM25 ranking                               │
    └────────────┬──────────────────────────────────┘
                 │
    ┌────────────▼──────────────────────────────┐
    │  DATA PROCESSING & CURATION (Week 5-8)    │
    ├───────────────────────────────────────────┤
    │ ETL Jobs (Scheduled):                     │
    │ 1. PubMed crawler (daily)                 │
    │ 2. PDF extraction (parallel, 12x)         │
    │ 3. Entity extraction + normalization      │
    │ 4. Relationship extraction (LLM)          │
    │ 5. Embedding generation                   │
    │ 6. Materialization (nightly)              │
    │                                           │
    │ Curation Pipeline:                        │
    │ • Entities pending → human approval       │
    │ • Relationships pending → human review    │
    │ • Only approved edges → KG                │
    │ • Feedback trains confidence models       │
    └───────────┬────────────────────────────────┘
                │
    ┌───────────▼────────────────────────────┐
    │   RAW DATA INGESTION (Week 4-5)        │
    ├───────────────────────────────────────┤
    │ PubMed API (esummary)                 │
    │ PMC Open Access XML                   │
    │ PDF URLs (Grobid + pymupdf)           │
    │ GEO/SRA datasets                      │
    │ Portfolio projects (AnnData + CSV)    │
    └───────────────────────────────────────┘
```

---

## Appendix B: File Structure

```
ebv-rag/
├── README.md                          # Start here
├── CONTRIBUTING.md
├── requirements.txt                   # Python dependencies
├── docker-compose.yml                 # LOCAL: dev setup
├── docker-compose.prod.yml            # PROD: deployment
├── Dockerfile
├── .github/
│   └── workflows/
│       ├── tests.yml                  # CI/CD
│       └── deploy.yml                 # Auto-deploy on merge
├── app/
│   ├── __init__.py
│   ├── main.py                        # FastAPI root
│   ├── config.py
│   ├── models.py                      # Pydantic schemas
│   ├── api/
│   │   ├── routes.py                  # Endpoints
│   │   └── dependencies.py            # Auth, rate limiting
│   ├── retrieval/
│   │   ├── hybrid.py                  # Orchestration
│   │   ├── bm25.py                    # SQLite FTS5
│   │   ├── vector.py                  # ChromaDB
│   │   ├── graph.py                   # Neo4j Cypher
│   │   └── ranker.py                  # Score merging
│   ├── synthesis/
│   │   ├── llm.py                     # Claude client
│   │   ├── prompts.py                 # Prompt templates
│   │   └── citation_extractor.py      # Citation parsing
│   ├── ingestion/
│   │   ├── pubmed_crawler.py
│   │   ├── pdf_extractor.py           # Hierarchical extraction
│   │   ├── entity_extractor.py        # NER + normalization
│   │   ├── relationship_extractor.py  # LLM-based
│   │   ├── embeddings_pipeline.py
│   │   └── validator.py               # Data quality checks
│   ├── curation/
│   │   ├── dashboard.py               # Streamlit UI
│   │   └── approval_workflow.py       # Curation logic
│   ├── database/
│   │   ├── postgres.py                # SQLAlchemy setup
│   │   ├── models.py                  # ORM models
│   │   └── crud.py                    # DB operations
│   ├── materialization/
│   │   ├── neo4j_builder.py           # Nightly rebuild
│   │   └── consistency_checker.py     # State validation
│   ├── scheduler/
│   │   └── run.py                     # Background job runner (SEPARATE PROCESS)
│   ├── evaluation/
│   │   ├── metrics.py                 # Precision, recall, NDCG
│   │   └── benchmark.py               # Eval harness
│   └── utils/
│       ├── cache.py
│       ├── logging_config.py
│       └── audit.py
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_retrieval.py
│   │   ├── test_entity_norm.py
│   │   └── test_confidence_scoring.py
│   ├── integration/
│   │   └── test_end_to_end.py
│   └── evaluation/
│       └── test_benchmark.py
├── data/
│   ├── benchmark/
│   │   └── queries.json               # Gold-standard eval set
│   ├── ontologies/                    # HGNC, Cell Ontology, DOID
│   └── sample_papers/                 # Test corpus
├── scripts/
│   ├── setup_postgres.sql
│   ├── setup_neo4j.cypher
│   ├── load_ontologies.py
│   └── export_backup.py
├── docs/
│   ├── ARCHITECTURE.md                # This file (v2.0)
│   ├── API.md
│   ├── DEPLOYMENT.md
│   ├── DATA_PIPELINE.md
│   ├── CURATION_WORKFLOW.md
│   └── KNOWLEDGE_GRAPH.md
└── .env.example
```

---

**END OF v2.0**

*Next revision (v2.1) should address: Neo4j materialization algorithm, specific Claude prompts, curation UX feedback, performance profiling, portfolio integration details.*

