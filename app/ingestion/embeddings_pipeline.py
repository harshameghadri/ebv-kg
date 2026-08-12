"""Chunk embedding pipeline that manages indexing document chunks from PostgreSQL to LanceDB."""

import logging
from typing import Any, Dict, List, Optional

from psycopg.rows import dict_row

from app.retrieval.embeddings import EmbeddingClient
from app.retrieval.vector import LanceDBClient

logger = logging.getLogger(__name__)


class EmbeddingsPipeline:
    """
    Pipeline to find unindexed document chunks in PostgreSQL,
    generate dense vector embeddings, and store them in LanceDB.
    """

    def __init__(
        self,
        embedding_client: Optional[EmbeddingClient] = None,
        vector_client: Optional[LanceDBClient] = None,
    ) -> None:
        """
        Initialize the EmbeddingsPipeline.

        Args:
            embedding_client: Optional EmbeddingClient. If None, a new default client is created.
            vector_client: Optional LanceDBClient. If None, a new default client is created.
        """
        self.embedding_client = embedding_client or EmbeddingClient()
        self.vector_client = vector_client or LanceDBClient()

    def index_pending_chunks(self, conn: Any, batch_size: int = 64, doc_ids: Optional[list[str]] = None) -> int:
        """
        Queries the PostgreSQL database for document chunks, filters out
        those already indexed in LanceDB, generates embeddings for the new ones,
        and saves them to LanceDB.

        Args:
            conn: A psycopg Connection object.
            batch_size: The batch size for processing embeddings and writes.
            doc_ids: Optional list of document IDs to process specifically.

        Returns:
            The total count of successfully indexed chunks.
        """
        # Ensure the LanceDB table is initialized
        self.vector_client.init_table()

        # Connect to LanceDB and fetch all currently indexed chunk IDs
        existing_ids = set()
        db = self.vector_client.connect()
        tables = db.list_tables()
        if not isinstance(tables, list):
            tables = getattr(tables, "tables", tables)

        if self.vector_client.table_name in tables:
            try:
                table = db.open_table(self.vector_client.table_name)
                arrow_table = table.to_arrow()
                if "id" in arrow_table.column_names:
                    existing_ids = {str(x) for x in arrow_table.column("id").to_pylist()}
            except Exception as e:
                logger.warning(f"Could not read existing IDs from LanceDB: {e}")

        # Fetch document chunks from PostgreSQL (filter by doc_ids if specified for instant execution)
        with conn.cursor(row_factory=dict_row) as cur:
            if doc_ids:
                cur.execute(
                    "SELECT c.id AS chunk_id, c.document_id, c.chunk_index, c.content, "
                    "d.pmid, d.doi, d.title "
                    "FROM document_chunks c "
                    "JOIN documents d ON c.document_id = d.id "
                    "WHERE c.document_id = ANY(%s)",
                    (doc_ids,)
                )
            else:
                cur.execute(
                    "SELECT c.id AS chunk_id, c.document_id, c.chunk_index, c.content, "
                    "d.pmid, d.doi, d.title "
                    "FROM document_chunks c "
                    "JOIN documents d ON c.document_id = d.id "
                    "ORDER BY c.id DESC LIMIT 2000"
                )
            rows = cur.fetchall()


        # Filter for chunks that have not yet been indexed
        pending_rows = []
        for row in rows:
            chunk_id_str = str(row["chunk_id"])
            if chunk_id_str not in existing_ids:
                pending_rows.append(row)

        total_indexed = 0
        if not pending_rows:
            return total_indexed

        # Process pending chunks in batches
        for i in range(0, len(pending_rows), batch_size):
            batch = pending_rows[i:i + batch_size]
            batch_texts = [row["content"] or "" for row in batch]

            # Generate dense vector embeddings for the batch
            embeddings = self.embedding_client.embed_documents(batch_texts)

            # Construct chunk representations for LanceDB
            lancedb_chunks = []
            for row, emb in zip(batch, embeddings):
                lancedb_chunks.append(
                    {
                        "id": str(row["chunk_id"]),
                        "document_id": str(row["document_id"]),
                        "chunk_index": row["chunk_index"],
                        "content": row["content"] or "",
                        "pmid": row["pmid"],
                        "doi": row["doi"],
                        "title": row["title"],
                        "vector": emb,
                    }
                )

            # Write the chunks to LanceDB
            self.vector_client.add_chunks(lancedb_chunks)
            total_indexed += len(batch)

        return total_indexed
