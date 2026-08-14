#!/bin/bash
cd /storage/harsha_projects/ebv_KG
export DATABASE_URL=postgresql://postgres:postgrespassword@localhost:5432/ebv_rag
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=neo4jpassword
export LANCEDB_URI=/storage/harsha_projects/ebv_KG/data/lancedb/
export EMBEDDINGS_MODEL=BAAI/bge-m3
export USE_LOCAL_LLM=true
exec /home/harsha/ebv_KG_venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8080
