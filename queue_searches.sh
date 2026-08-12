#!/bin/bash

# List of 51 targeted search terms for Epstein-Barr Virus literature scraping
QUERIES=(
    "Epstein-Barr virus"
    "Epstein-Barr virus LMP1"
    "Epstein-Barr virus LMP2"
    "Epstein-Barr virus EBNA1"
    "Epstein-Barr virus EBNA2"
    "Epstein-Barr virus EBNA3"
    "Epstein-Barr virus EBER"
    "EBV gp350"
    "EBV BZLF1"
    "EBV lytic replication"
    "EBV latency"
    "EBV B-cell transformation"
    "EBV Burkitt lymphoma"
    "EBV Hodgkin lymphoma"
    "EBV Nasopharyngeal carcinoma"
    "EBV Gastric carcinoma"
    "EBV post-transplant lymphoproliferative disorder"
    "EBV PTLD"
    "EBV mononucleosis"
    "EBV CD21"
    "EBV glycoprotein gH gL"
    "EBV glycoprotein gB"
    "EBV envelope gp42"
    "EBV immortalization"
    "EBV vaccine"
    "EBV monoclonal antibodies"
    "EBV acyclovir"
    "EBV ganciclovir"
    "LMP1 NF-kB pathway"
    "LMP1 TRAF interaction"
    "LMP2A ITAM signaling"
    "EBNA1 DNA binding"
    "EBNA2 Notch signaling"
    "EBNA3C cell cycle"
    "EBER1 EBER2 B-cells"
    "EBV BART miRNA"
    "EBV BHRF1 Bcl-2"
    "EBV BALF5 DNA polymerase"
    "EBV BCRF1 IL-10"
    "EBV BARF1 oncogene"
    "EBV BRLF1 Rta"
    "EBV HLA class II"
    "EBV integrin binding"
    "EBV-associated HLH"
    "EBV LMP1 CD40 mimicry"
    "EBV super-enhancers"
    "EBV EBI3 IL-27"
    "Oral hairy leukoplakia EBV"
    "EBV primary infection"
    "EBV persistent infection"
    "EBV lytic switch"
)

PYTHON_BIN="/home/harsha/ebv_KG_venv/bin/python"
PIPELINE_SCRIPT="/storage/harsha_projects/ebv_KG/run_pipeline.py"
STAGING_DIR="/storage/harsha_projects/ebv_KG/data/staging"

PG_DSN="postgresql://postgres:postgrespassword@localhost:5432/ebv_rag"
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASS="neo4jpassword"
LANCEDB_URI="/storage/harsha_projects/ebv_KG/data/lancedb/"

# Initialize the database schemas and indices synchronously before enqueuing to prevent parallel initialization race conditions
echo "Initializing database schemas and search indices..."
$PYTHON_BIN $PIPELINE_SCRIPT --query "init" --max-articles 0 --staging-dir $STAGING_DIR --pg-dsn $PG_DSN --neo4j-uri $NEO4J_URI --neo4j-user $NEO4J_USER --neo4j-password $NEO4J_PASS --lancedb-uri $LANCEDB_URI

echo "Queuing ${#QUERIES[@]} search terms in pueue..."

# Enqueue tasks into dedicated dbingest group
PUEUE_BIN="/storage/harsha_projects/server_environments/bin/pueue"

# Ensure group dbingest exists and has parallel 20 worker limit
$PUEUE_BIN group add dbingest 2>/dev/null || true
$PUEUE_BIN parallel 20 -g dbingest 2>/dev/null || true

for QUERY in "${QUERIES[@]}"; do
    echo "Queuing query in dbingest: '$QUERY'"
    
    # Pass --skip-init since all schemas and indices have already been initialized
    $PUEUE_BIN add -g dbingest -- "$PYTHON_BIN $PIPELINE_SCRIPT --query \"$QUERY\" --max-articles 1000 --staging-dir $STAGING_DIR --pg-dsn $PG_DSN --neo4j-uri $NEO4J_URI --neo4j-user $NEO4J_USER --neo4j-password $NEO4J_PASS --lancedb-uri $LANCEDB_URI --skip-init"
done

echo "All jobs enqueued successfully in Pueue group 'dbingest'."

