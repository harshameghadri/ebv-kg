"""LLM Synthesis Client using Anthropic Claude."""

import json
import os
import re

from anthropic import Anthropic

from app.synthesis.prompts import SYSTEM_PROMPT_TEMPLATE, USER_PROMPT_TEMPLATE


class ClaudeSynthesisClient:
    """Client for generating synthesized answers using Claude."""
    
    def __init__(
        self, api_key: str = None, model: str = "claude-3-5-sonnet-20240620"
    ):
        """Initialize the client with an optional API key and model."""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        # Initialize Anthropic client. If api_key is None, standard client raises.
        # To make testing easy, use dummy key if api_key and env are missing.
        self.client = Anthropic(api_key=self.api_key or "dummy_key")

    def synthesize(
        self, query: str, retrieved_chunks: list[dict], graph_context: str = ""
    ) -> dict:
        """Synthesize a query and retrieved chunks/graph context into an answer.
        
        Returns:
            dict: {"answer": str, "confidence": float, "citations": list[dict]}
        """
        # If no chunks and no graph context, return early
        if not retrieved_chunks and not graph_context:
            return {
                "answer": "I do not know",
                "confidence": 0.0,
                "citations": []
            }
            
        # Format the document chunks
        formatted_chunks = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            chunk_id = chunk.get("id") or chunk.get("chunk_id")
            content = chunk.get("content") or chunk.get("text") or ""
            pmid = chunk.get("pmid")
            doi = chunk.get("doi")
            title = chunk.get("title")
            
            # Check nested metadata
            metadata = chunk.get("metadata")
            if isinstance(metadata, dict):
                chunk_id = (
                    chunk_id
                    or metadata.get("id")
                    or metadata.get("chunk_id")
                )
                pmid = pmid or metadata.get("pmid")
                doi = doi or metadata.get("doi")
                title = title or metadata.get("title")
                
            chunk_id = chunk_id or f"chunk-{idx}"
            pmid = pmid or "N/A"
            doi = doi or "N/A"
            title = title or "N/A"
            
            formatted_chunks.append(
                f"[{idx}] Chunk ID: {chunk_id}\n"
                f"Title: {title}\n"
                f"PMID: {pmid} | DOI: {doi}\n"
                f"Content: {content}\n"
                "---"
            )
            
        document_chunks_str = (
            "\n".join(formatted_chunks)
            if formatted_chunks
            else "No retrieved chunks."
        )
        graph_context_str = (
            graph_context if graph_context else "No graph context provided."
        )
        
        # Build user prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            query=query,
            document_chunks=document_chunks_str,
            graph_context=graph_context_str
        )
        
        try:
            # Call Anthropic API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT_TEMPLATE,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            response_text = response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Failed to call Anthropic API: {e}") from e
            
        # Parse the response
        try:
            # Look for JSON structure
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON object found in response.")
                
            parsed = json.loads(json_match.group(0))
            answer = parsed.get("answer", "I do not know")
            
            # Confidence
            confidence = parsed.get("confidence", 0.0)
            try:
                confidence = float(confidence)
            except (ValueError, TypeError):
                confidence = 0.0
                
            # Extract citations
            citation_indices = set()
            matches = re.findall(r"\[(\d+)\]", answer)
            for m in matches:
                try:
                    citation_indices.add(int(m))
                except ValueError:
                    pass
                    
            # Also extract from the returned citations field if present
            returned_citations = parsed.get("citations", [])
            if isinstance(returned_citations, list):
                for cit in returned_citations:
                    if isinstance(cit, dict):
                        for key in [
                            "source_index",
                            "citation_index",
                            "id",
                            "citation_id",
                        ]:
                            val = cit.get(key)
                            if val is not None:
                                try:
                                    cleaned = (
                                        str(val)
                                        .replace("[", "")
                                        .replace("]", "")
                                    )
                                    citation_indices.add(int(cleaned))
                                except ValueError:
                                    pass
                                    
            citations = []
            for idx in sorted(citation_indices):
                if 1 <= idx <= len(retrieved_chunks):
                    chunk = retrieved_chunks[idx - 1]
                    chunk_id = chunk.get("id") or chunk.get("chunk_id")
                    pmid = chunk.get("pmid")
                    doi = chunk.get("doi")
                    
                    # Nested metadata check
                    metadata = chunk.get("metadata")
                    if isinstance(metadata, dict):
                        chunk_id = (
                            chunk_id
                            or metadata.get("id")
                            or metadata.get("chunk_id")
                        )
                        pmid = pmid or metadata.get("pmid")
                        doi = doi or metadata.get("doi")
                        
                    chunk_id = chunk_id or f"chunk-{idx}"
                    pmid = pmid or "N/A"
                    doi = doi or "N/A"
                    
                    citations.append({
                        "source_index": idx,
                        "chunk_id": chunk_id,
                        "pmid": pmid,
                        "doi": doi
                    })
                    
            return {
                "answer": answer,
                "confidence": confidence,
                "citations": citations
            }
            
        except Exception as e:
            raise ValueError(
                f"Failed to parse Claude response: {e}. "
                f"Raw response: {response_text}"
            ) from e
