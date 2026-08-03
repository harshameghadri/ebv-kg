"""LanceDB Vector Store Client for storing and querying document chunk embeddings."""

import os
import logging
from typing import Any, Dict, List, Optional
import lancedb
import pyarrow as pa

logger = logging.getLogger(__name__)


class LanceDBClient:
    """Wrapper client for managing and querying LanceDB vector database tables."""

    def __init__(
        self,
        uri: Optional[str] = None,
        table_name: str = "chunks",
        vector_dim: int = 1024,
    ) -> None:
        """Initialize LanceDBClient.

        Args:
            uri: Path or URI to the LanceDB database. Defaults to environment variable
                LANCEDB_URI or 'data/lancedb/'.
            table_name: Name of the table. Defaults to 'chunks'.
            vector_dim: Dimension of the dense vector embeddings. Defaults to 1024.
        """
        self.uri = uri or os.getenv("LANCEDB_URI", "data/lancedb/")
        self.table_name = table_name
        self.vector_dim = vector_dim
        self._db: Optional[lancedb.DBConnection] = None
        self._table: Optional[lancedb.table.Table] = None

    def connect(self) -> lancedb.DBConnection:
        """Establish a connection to the LanceDB database.

        Returns:
            lancedb.DBConnection: Connection object.
        """
        if self._db is None:
            # Ensure the directory exists if it's a local path
            if (
                not self.uri.startswith("memory://")
                and not self.uri.startswith("db://")
                and not self.uri.startswith("s3://")
            ):
                os.makedirs(self.uri, exist_ok=True)
            self._db = lancedb.connect(self.uri)
        return self._db

    def init_table(self) -> lancedb.table.Table:
        """Idempotently creates the LanceDB table.

        Returns:
            lancedb.table.Table: The opened or created table.
        """
        db = self.connect()
        tables = db.list_tables()
        if not isinstance(tables, list):
            tables = getattr(tables, "tables", tables)

        if self.table_name in tables:
            try:
                table = db.open_table(self.table_name)
                existing_schema = table.schema
                if "vector" in existing_schema.names:
                    vector_field = existing_schema.field("vector")
                    list_size = getattr(vector_field.type, "list_size", None)
                    if list_size is None and hasattr(vector_field.type, "value_length"):
                        list_size = vector_field.type.value_length
                    
                    if list_size is not None and list_size != self.vector_dim:
                        logger.warning(
                            "LanceDB table vector dimension mismatch (existing: %s, requested: %s). Dropping table to recreate.",
                            str(list_size),
                            str(self.vector_dim)
                        )
                        db.drop_table(self.table_name)
            except Exception as e:
                logger.warning("Error checking existing table schema: %s", e)

        # Define schema for chunks using pyarrow
        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("document_id", pa.string()),
                pa.field("chunk_index", pa.int32()),
                pa.field("content", pa.string()),
                pa.field("pmid", pa.string(), nullable=True),
                pa.field("doi", pa.string(), nullable=True),
                pa.field("title", pa.string(), nullable=True),
                pa.field("vector", pa.list_(pa.float32(), self.vector_dim)),
            ]
        )

        self._table = db.create_table(self.table_name, schema=schema, exist_ok=True)
        return self._table

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Adds chunks to the vector store.

        Chunks contain text content, document metadata (pmid, doi, title),
        and their generated dense embeddings.

        Args:
            chunks: A list of dictionaries representing document chunks.
        """
        if self._table is None:
            self.init_table()

        processed_chunks = []
        for chunk in chunks:
            # Extract core fields
            chunk_id = chunk.get("id")
            doc_id = chunk.get("document_id")
            chunk_index = chunk.get("chunk_index")
            content = chunk.get("content")
            vector = chunk.get("vector")

            # Validate core fields
            if vector is None:
                raise ValueError("Each chunk must contain a 'vector' field")

            # Extract metadata fields (could be nested under "metadata" key or flat)
            metadata = chunk.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}

            pmid = chunk.get("pmid") or metadata.get("pmid")
            doi = chunk.get("doi") or metadata.get("doi")
            title = chunk.get("title") or metadata.get("title")

            processed_chunks.append(
                {
                    "id": str(chunk_id) if chunk_id is not None else "",
                    "document_id": str(doc_id) if doc_id is not None else "",
                    "chunk_index": int(chunk_index) if chunk_index is not None else 0,
                    "content": str(content) if content is not None else "",
                    "pmid": str(pmid) if pmid is not None else None,
                    "doi": str(doi) if doi is not None else None,
                    "title": str(title) if title is not None else None,
                    "vector": [float(x) for x in vector],
                }
            )

        if processed_chunks:
            self._table.add(processed_chunks)

    def search_vector(
        self, vector: List[float], limit: int = 10, metric: str = "l2"
    ) -> List[Dict[str, Any]]:
        """Performs vector search returning matched chunks with similarity scores and metadata.

        Args:
            vector: The query vector.
            limit: Maximum number of records to return. Defaults to 10.
            metric: Distance metric to use ('l2', 'cosine', or 'dot'). Defaults to 'l2'.

        Returns:
            List of dictionaries representing the matched chunks with similarity scores.
        """
        if self._table is None:
            self.init_table()

        # Execute search query
        query = self._table.search(vector).metric(metric).limit(limit)
        results = query.to_list()

        output = []
        for item in results:
            res_dict = dict(item)

            # Retrieve _distance
            distance = res_dict.get("_distance", 0.0)

            # Compute a similarity score based on metric
            if metric in ("cosine", "dot"):
                # Cosine/Dot distance = 1 - similarity
                similarity = 1.0 - distance
            else:  # l2
                # Convert L2 distance to a similarity score in range (0, 1]
                similarity = 1.0 / (1.0 + distance)

            res_dict["score"] = similarity
            output.append(res_dict)

        return output

    def clear_table(self) -> None:
        """Drops the table or clears all vectors."""
        db = self.connect()
        try:
            db.drop_table(self.table_name)
        except Exception:
            # If table does not exist or another error, we ignore it to be idempotent/safe
            pass
        self._table = None
