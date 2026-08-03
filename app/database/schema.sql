-- PostgreSQL Database Schema Definition for EBV Knowledge System

-- 1. Documents Table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    doi VARCHAR UNIQUE,
    pmid VARCHAR,
    title TEXT,
    journal VARCHAR,
    published_date DATE,
    parsed_json JSONB
);

-- Indexes for documents
CREATE INDEX IF NOT EXISTS idx_documents_doi ON documents(doi);
CREATE INDEX IF NOT EXISTS idx_documents_pmid ON documents(pmid);

-- 2. Document Chunks Table
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content TEXT,
    token_count INT,
    CONSTRAINT uq_document_chunks_doc_index UNIQUE (document_id, chunk_index)
);

-- Indexes for document_chunks
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);

-- 3. Normalized Entities Table
CREATE TABLE IF NOT EXISTS normalized_entities (
    id UUID PRIMARY KEY,
    canonical_id VARCHAR UNIQUE,
    name VARCHAR,
    entity_type VARCHAR,
    ontology_source VARCHAR,
    synonyms TEXT[]
);

-- Indexes for normalized_entities
CREATE INDEX IF NOT EXISTS idx_normalized_entities_canonical_id ON normalized_entities(canonical_id);
CREATE INDEX IF NOT EXISTS idx_normalized_entities_name ON normalized_entities(name);

-- 4. Relationships Table
CREATE TABLE IF NOT EXISTS relationships (
    id UUID PRIMARY KEY,
    source_entity_id UUID NOT NULL REFERENCES normalized_entities(id) ON DELETE CASCADE,
    target_entity_id UUID NOT NULL REFERENCES normalized_entities(id) ON DELETE CASCADE,
    relationship_type VARCHAR NOT NULL,
    confidence_score DOUBLE PRECISION,
    curation_status VARCHAR,
    source_type VARCHAR,
    CONSTRAINT uq_relationships_source_target_type UNIQUE (source_entity_id, target_entity_id, relationship_type)
);

-- Indexes for relationships
CREATE INDEX IF NOT EXISTS idx_relationships_source_entity_id ON relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target_entity_id ON relationships(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_source_target ON relationships(source_entity_id, target_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_curation_status ON relationships(curation_status);
CREATE INDEX IF NOT EXISTS idx_relationships_curation_confidence ON relationships(curation_status, confidence_score);

-- 5. Relationship Evidence Table
CREATE TABLE IF NOT EXISTS relationship_evidence (
    id UUID PRIMARY KEY,
    relationship_id UUID NOT NULL REFERENCES relationships(id) ON DELETE CASCADE,
    chunk_id UUID NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    confidence_score DOUBLE PRECISION,
    citation_text TEXT,
    CONSTRAINT uq_evidence_rel_chunk UNIQUE (relationship_id, chunk_id)
);

-- Indexes for relationship_evidence
CREATE INDEX IF NOT EXISTS idx_relationship_evidence_relationship_id ON relationship_evidence(relationship_id);
CREATE INDEX IF NOT EXISTS idx_relationship_evidence_chunk_id ON relationship_evidence(chunk_id);
