# EBV Knowledge System 🧬📊

> **A Multi-Scale Knowledge Graph & Retrieval-Augmented Generation (Graph-RAG) Platform for Epstein-Barr Virus Research**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![LanceDB](https://img.shields.io/badge/LanceDB-VectorStore-orange.svg)](https://lancedb.github.io/lancedb/)
[![Neo4j](https://img.shields.io/badge/Neo4j-GraphDB-45818e.svg)](https://neo4j.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Executive Summary & Motivation

Epstein-Barr Virus (EBV) research is historically fragmented across isolated biological domain silos:
* **Virology**: Mechanistic viral gene regulation (e.g., `LMP1`, `EBNA2`, `EBNA3C`).
* **Oncology**: Pathogenesis of Burkitt Lymphoma (BL), Nasopharyngeal Carcinoma (NPC), and Hodgkin Lymphoma.
* **Immunology**: Latent membrane proteins and B cell fate (e.g., Atypical B cells, Germinal Center niches).
* **Neurology**: Strong epidemiologic and mechanistic links to Multiple Sclerosis (MS).

The **EBV Knowledge System** bridges these research silos by serving as a unified, queryable reference layer. It integrates multi-scale biomedical data—from full-text scientific literature and single-cell RNA-seq marker datasets to structured ontology graphs—into a production-grade **Graph-RAG platform**.

---

## 🚀 Key Features

* **Multi-Scale Knowledge Schema**: Connects 7 canonical entity types (`GENE`, `PROTEIN`, `CELL_TYPE`, `TISSUE`, `DISEASE_OUTCOME`, `PHENOTYPE`, `CHEMICAL_COMPOUND`) across 8 structured relationship predicates (`EXPRESSES`, `INHIBITS`, `ACTIVATES`, `INTERACTS_WITH`, `IS_MARKER_FOR`, `ASSOCIATED_WITH`, `LOCATED_IN`, `TARGETS`).
* **Hybrid Subgraph Vector RAG**: Combines dense semantic vector retrieval (**LanceDB** with `allenai/specter2` embeddings) with multi-hop graph traversal.
* **SPOKE-Inspired Path Pruning**: Features a vector-similarity 2-hop neighborhood pruner (`SubgraphPruner`) and path-to-text fact serializer (`FactSerializer`) to fit long graph contexts into LLM prompts without noise.
* **Single-Cell Omics Funnel**: Includes an `AnnData` parser (`.h5ad` & marker CSVs) that extracts cell-type marker genes and calculates edge confidence from $log_2\text{FC}$ and $p_{\text{adj}}$ values.
* **Cross-Silo Hypothesis Engine**: Surfaces shared biological niches and cellular phenotypes (e.g., $TBX21^+$ Atypical B cells overlapping between Multiple Sclerosis and Burkitt Lymphoma).
* **Pluggable Graph Engines**: Built-in support for **Neo4j** (production graph DB), embedded C++ **KùzuDB** (zero-dependency local execution), and **PostgreSQL Apache AGE** (relational graph extension).
* **Interactive Visual Dashboard**: Includes a responsive web interface powered by **Cytoscape.js** for visual 2-hop subgraph navigation and dual literature provenance citations.

---

## 📐 System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │      Literature & Multi-Omics Sources        │
                    │   (PubMed JATS XML, Single-Cell Omics)       │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │     Ingestion & Bio-NER Extraction           │
                    │  (SciSpacy / BERN2 Entity Normalization)     │
                    └──────────────────────┬───────────────────────┘
                                           │
                    ┌──────────────────────┴───────────────────────┐
                    │                                              │
                    ▼                                              ▼
    ┌───────────────────────────────┐              ┌───────────────────────────────┐
    │     Dense Vector Store        │              │  Relational & Graph Storage   │
    │  (LanceDB + specter2 Embeds)  │              │ (PostgreSQL + Neo4j / KùzuDB) │
    └───────────────┬───────────────┘              └───────────────┬───────────────┘
                    │                                              │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │           Graph-RAG Subgraph Engine          │
                    │  (2-Hop Pruning -> Fact Serialization)       │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │      FastAPI Server & Visual UI Dashboard    │
                    │  (Cytoscape.js Graph View & Dual Citations)  │
                    └──────────────────────────────────────────────┘
```

---

## 📊 Performance & Scale

The live system operates at scale on **rinamochana**:

| System Component | Volume / Metric | Description |
| :--- | :--- | :--- |
| **Ingested Documents** | **13,040+ papers** | PMC JATS XML full-text articles & PubMed metadata |
| **Text Chunks** | **231,900+ sections** | 768-dimensional embedded text sections in LanceDB |
| **Normalized Entities** | **24,900+ biological nodes** | HGNC Genes, UniProt Proteins, Cell Ontology, MeSH Diseases |
| **Extracted Relationships** | **739,000+ edges** | Contextual bio-entity interactions & co-occurrences |
| **Auto-Approved KG Edges** | **1,500+ high-conf edges** | Promoted directly to Neo4j graph for multi-hop RAG ($\ge 0.80$ conf) |

---

## 🛠️ Quickstart Guide

### Prerequisites
* **Python**: 3.10 or higher
* **PostgreSQL**: 14+ (or Docker)
* **Neo4j**: 5.x (Optional for production graph store)

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/harshameghadri/ebv-kg.git
cd ebv-kg

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the root directory:
```env
DATABASE_URL="postgresql://postgres:postgrespassword@localhost:5432/ebv_rag"
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="neo4jpassword"
LANCEDB_URI="data/lancedb/"
HF_TOKEN="your_huggingface_token_here"
```

### 3. Run the Web Application & REST API
Start the FastAPI server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Open your browser and navigate to:
* **Interactive UI Dashboard**: `http://localhost:8000/`
* **OpenAPI Documentation**: `http://localhost:8000/docs`

---

## 💻 Usage & CLI Tools

### Execute Graph-RAG Pipeline
Run an end-to-end multi-hop Graph-RAG query from the command line:
```bash
python run_pipeline.py --query "How does EBV LMP1 interact with TRAF proteins in B cell transformation?"
```

### Ingest Single-Cell Omics Datasets
Parse single-cell RNA-seq marker datasets into the knowledge graph:
```bash
python -m app.ingestion.anndata_cli \
  --input data/staging/markers.csv \
  --cluster-key cell_type \
  --pg-dsn postgresql://postgres:postgrespassword@localhost:5432/ebv_rag
```

### Benchmark Graph Engines
Compare write throughput and read latency across Neo4j, KùzuDB, and Apache AGE:
```bash
python -m app.materialization.benchmark_graph_engines --nodes 1000 --edges 5000
```

---

## 🌐 REST API Specification

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/v1/search` | `POST` | Executes hybrid semantic vector + multi-hop graph RAG search. |
| `/api/v1/hypothesis/niche-overlap` | `POST` | Identifies shared cellular states & marker genes across disease outcomes. |
| `/api/v1/graph` | `GET` | Retrieves 2-hop neighborhood graph paths for Cytoscape.js rendering. |
| `/api/v1/health` | `GET` | System health check for PostgreSQL, LanceDB, Neo4j, and KùzuDB. |
| `/api/v1/metrics` | `GET` | Real-time counts of ingested documents, chunks, entities, and edges. |

---

## 🧪 Running Unit Tests

The test suite contains **224 unit tests** with 100% pass rate:
```bash
pytest tests/ -v
```

---

## 📜 License

This project is licensed under the **[PolyForm Noncommercial License 1.0.0](LICENSE)**. Personal, educational, and academic research use is permitted; commercial use is restricted.

