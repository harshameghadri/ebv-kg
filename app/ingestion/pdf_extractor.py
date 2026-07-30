"""PDF Extractor module using Grobid with PyMuPDF fallback.

Extracts structured metadata, section-based text chunks, and references from PDF files
via a self-hosted Grobid service with an automatic fallback to PyMuPDF (fitz).
"""

import collections
import logging
import os
import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import requests

try:
    from lxml import etree
except ImportError:
    etree = None

import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


def _local_name(tag: Any) -> str:
    """Return local name of XML tag, stripping namespace if present."""
    if isinstance(tag, str):
        if "}" in tag:
            return tag.split("}", 1)[1]
        return tag
    return ""


def _find_all_local(parent: Any, tag_name: str) -> list[Any]:
    """Find all descendant elements matching tag_name (namespace-agnostic)."""
    return [elem for elem in parent.iter() if _local_name(elem.tag) == tag_name]


def _find_child_local(parent: Any, tag_name: str) -> Any | None:
    """Find first direct child element matching tag_name (namespace-agnostic)."""
    for child in parent:
        if _local_name(child.tag) == tag_name:
            return child
    return None


def _find_children_local(parent: Any, tag_name: str) -> list[Any]:
    """Find all direct child elements matching tag_name (namespace-agnostic)."""
    return [child for child in parent if _local_name(child.tag) == tag_name]


def _find_descendant_local(parent: Any, tag_name: str) -> Any | None:
    """Find first descendant matching tag_name (namespace-agnostic)."""
    for elem in parent.iter():
        if elem is not parent and _local_name(elem.tag) == tag_name:
            return elem
    return None


def _get_text(elem: Any | None) -> str:
    """Extract and normalize all inner text of an element, including child elements."""
    if elem is None:
        return ""
    text_parts = list(elem.itertext())
    joined = "".join(text_parts)
    return " ".join(joined.split())


def _parse_xml_bytes(content: bytes) -> Any:
    """Parse raw XML bytes into element tree root."""
    if etree is not None:
        try:
            return etree.fromstring(content)
        except etree.XMLSyntaxError as e:
            raise ValueError(f"Invalid XML content provided: {e}") from e
    else:
        try:
            return ET.fromstring(content)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML content provided: {e}") from e


class PDFExtractor:
    """PDF Extractor using Grobid TEI/XML parser with PyMuPDF fallback."""

    def __init__(self, grobid_url: str = "http://localhost:8070", timeout: float = 10.0):
        """Initialize PDFExtractor.

        Args:
            grobid_url: URL of self-hosted Grobid service.
            timeout: Timeout in seconds for Grobid HTTP request.
        """
        self.grobid_url = grobid_url.rstrip("/")
        self.timeout = timeout

    def parse(self, pdf_input: str | bytes | Path) -> dict[str, Any]:
        """Extract structured content from a PDF file or bytes.

        Args:
            pdf_input: Path object, file path string, or raw PDF bytes.

        Returns:
            Dict containing 'metadata', 'chunks', 'text_chunks', and 'references'.
        """
        pdf_bytes = self._read_pdf_input(pdf_input)

        # 1. Attempt extraction via Grobid
        grobid_result = self._extract_via_grobid(pdf_bytes)
        if grobid_result is not None:
            return grobid_result

        # 2. Fall back to PyMuPDF
        logger.info("Grobid service unavailable or failed. Falling back to PyMuPDF.")
        return self._extract_via_pymupdf(pdf_bytes, pdf_input=pdf_input)

    def _read_pdf_input(self, pdf_input: str | bytes | Path) -> bytes:
        """Resolve pdf_input to raw bytes."""
        if isinstance(pdf_input, Path):
            if not pdf_input.exists():
                raise FileNotFoundError(f"File not found: {pdf_input}")
            return pdf_input.read_bytes()
        elif isinstance(pdf_input, str):
            if os.path.isfile(pdf_input):
                return Path(pdf_input).read_bytes()
            elif pdf_input.startswith("%PDF"):
                return pdf_input.encode("latin1")
            else:
                raise FileNotFoundError(f"File not found: {pdf_input}")
        elif isinstance(pdf_input, bytes):
            if not pdf_input.startswith(b"%PDF") and len(pdf_input) == 0:
                raise ValueError("Empty or invalid PDF content provided.")
            return pdf_input
        else:
            raise TypeError(f"Unsupported pdf_input type: {type(pdf_input)}")

    def _extract_via_grobid(self, pdf_bytes: bytes) -> dict[str, Any] | None:
        """Send PDF to Grobid service and parse response."""
        endpoint = f"{self.grobid_url}/api/processFulltextDocument"
        files = {"input": ("document.pdf", pdf_bytes, "application/pdf")}
        data = {"consolidateHeader": "1", "consolidateCitations": "1"}

        try:
            response = requests.post(
                endpoint, files=files, data=data, timeout=self.timeout
            )
            if response.status_code != 200:
                logger.warning(
                    f"Grobid returned status code {response.status_code}: {response.text[:200]}"
                )
                return None
            return self._parse_grobid_tei(response.content)
        except (requests.RequestException, ValueError, Exception) as e:
            logger.warning(f"Grobid processing failed: {e}")
            return None

    def _parse_grobid_tei(self, tei_xml_bytes: bytes) -> dict[str, Any]:
        """Parse TEI/XML content returned by Grobid."""
        root = _parse_xml_bytes(tei_xml_bytes)

        metadata = self._extract_grobid_metadata(root)
        chunks = self._extract_grobid_chunks(root)
        references = self._extract_grobid_references(root)

        return {
            "metadata": metadata,
            "chunks": chunks,
            "text_chunks": chunks,
            "references": references,
        }

    def _extract_grobid_metadata(self, root: Any) -> dict[str, Any]:
        """Extract article metadata from TEI XML header."""
        header = _find_descendant_local(root, "teiHeader")
        if header is None:
            header = root

        title: str | None = None
        authors: list[str] = []
        doi: str | None = None
        pmid: str | None = None
        pmcid: str | None = None
        journal: str | None = None
        pub_date: str | None = None
        year: str | None = None

        # Title
        title_stmt = _find_descendant_local(header, "titleStmt")
        if title_stmt is not None:
            for t_elem in _find_all_local(title_stmt, "title"):
                t_type = t_elem.attrib.get("type", "")
                if t_type == "main" or not title:
                    title = _get_text(t_elem)
                    if t_type == "main":
                        break
        if not title:
            analytic = _find_descendant_local(header, "analytic")
            if analytic is not None:
                title_elem = _find_child_local(analytic, "title")
                if title_elem is not None:
                    title = _get_text(title_elem)

        # Authors
        for author_elem in _find_all_local(header, "author"):
            pers_elem = _find_child_local(author_elem, "persName")
            if pers_elem is not None:
                forenames = [
                    _get_text(fn)
                    for fn in _find_children_local(pers_elem, "forename")
                    if _get_text(fn)
                ]
                surname_elem = _find_child_local(pers_elem, "surname")
                surname = _get_text(surname_elem) if surname_elem is not None else ""
                if forenames and surname:
                    authors.append(f"{' '.join(forenames)} {surname}")
                elif surname:
                    authors.append(surname)
                elif forenames:
                    authors.append(" ".join(forenames))
            else:
                author_text = _get_text(author_elem)
                if author_text:
                    authors.append(author_text)

        # Deduplicate authors preserving order
        unique_authors = list(dict.fromkeys(authors))

        # IDs (DOI, PMID, PMCID)
        for idno in _find_all_local(header, "idno"):
            id_type = str(idno.attrib.get("type", "")).lower()
            val = _get_text(idno)
            if val:
                if id_type == "doi":
                    doi = val
                elif id_type == "pmid":
                    pmid = val
                elif id_type in ("pmc", "pmcid"):
                    pmcid = val

        # Journal
        monogr = _find_descendant_local(header, "monogr")
        if monogr is not None:
            for j_title in _find_all_local(monogr, "title"):
                j_level = j_title.attrib.get("level", "")
                if j_level == "j" or not journal:
                    journal = _get_text(j_title)
                    if j_level == "j":
                        break

        # Publication Date / Year
        imprint = _find_descendant_local(header, "imprint")
        date_source = imprint if imprint is not None else header
        date_elem = _find_descendant_local(date_source, "date")
        if date_elem is not None:
            when = date_elem.attrib.get("when", "")
            date_text = _get_text(date_elem)
            pub_date = when if when else date_text
            if pub_date:
                match = re.search(r"\b(19\d{2}|20\d{2})\b", pub_date)
                if match:
                    year = match.group(1)
                else:
                    year = pub_date[:4] if pub_date[:4].isdigit() else None

        return {
            "title": title,
            "authors": unique_authors,
            "doi": doi,
            "pmid": pmid,
            "pmcid": pmcid,
            "journal": journal,
            "publication_date": pub_date,
            "year": year,
        }

    def _extract_grobid_chunks(self, root: Any) -> list[dict[str, str]]:
        """Extract body text chunks and abstract from TEI XML."""
        chunks: list[dict[str, str]] = []

        # 1. Abstract
        abstract_elem = _find_descendant_local(root, "abstract")
        if abstract_elem is not None:
            for p in _find_all_local(abstract_elem, "p"):
                txt = _get_text(p)
                if txt:
                    chunks.append({"section": "Abstract", "text": txt})

        # 2. Body
        body = _find_descendant_local(root, "body")
        if body is not None:
            def process_div(div_elem: Any, path: list[str]) -> None:
                head_elem = _find_child_local(div_elem, "head")
                sec_name = _get_text(head_elem) if head_elem is not None else ""
                current_path = list(path)
                if sec_name:
                    current_path.append(sec_name)

                section_str = " > ".join(current_path) if current_path else "Body"

                for child in div_elem:
                    tag = _local_name(child.tag)
                    if tag == "p":
                        txt = _get_text(child)
                        if txt:
                            chunks.append({"section": section_str, "text": txt})
                    elif tag == "div":
                        process_div(child, current_path)

            divs = _find_children_local(body, "div")
            if divs:
                for div in divs:
                    process_div(div, [])
            else:
                for p in _find_all_local(body, "p"):
                    txt = _get_text(p)
                    if txt:
                        chunks.append({"section": "Body", "text": txt})

        return chunks

    def _extract_grobid_references(self, root: Any) -> list[dict[str, Any]]:
        """Extract bibliography references from TEI XML."""
        references: list[dict[str, Any]] = []

        list_bibl = _find_descendant_local(root, "listBibl")
        ref_source = list_bibl if list_bibl is not None else root

        for bibl in _find_all_local(ref_source, "biblStruct"):
            ref_title: str | None = None
            ref_authors: list[str] = []
            ref_journal: str | None = None
            ref_year: str | None = None
            ref_doi: str | None = None

            analytic = _find_child_local(bibl, "analytic")
            monogr = _find_child_local(bibl, "monogr")

            # Title
            if analytic is not None:
                title_elem = _find_child_local(analytic, "title")
                if title_elem is not None:
                    ref_title = _get_text(title_elem)
            if not ref_title and monogr is not None:
                title_elem = _find_child_local(monogr, "title")
                if title_elem is not None:
                    ref_title = _get_text(title_elem)

            # Authors
            author_parent = analytic if analytic is not None else bibl
            for author in _find_all_local(author_parent, "author"):
                pers = _find_child_local(author, "persName")
                if pers is not None:
                    fns = [
                        _get_text(fn)
                        for fn in _find_children_local(pers, "forename")
                        if _get_text(fn)
                    ]
                    sn_elem = _find_child_local(pers, "surname")
                    sn = _get_text(sn_elem) if sn_elem is not None else ""
                    if fns and sn:
                        ref_authors.append(f"{' '.join(fns)} {sn}")
                    elif sn:
                        ref_authors.append(sn)
                    elif fns:
                        ref_authors.append(" ".join(fns))
                else:
                    txt = _get_text(author)
                    if txt:
                        ref_authors.append(txt)

            # Journal
            if monogr is not None:
                for j_elem in _find_all_local(monogr, "title"):
                    if j_elem.attrib.get("level") == "j" or not ref_journal:
                        ref_journal = _get_text(j_elem)

            # Year
            date_elem = _find_descendant_local(bibl, "date")
            if date_elem is not None:
                when = date_elem.attrib.get("when", "")
                date_txt = when or _get_text(date_elem)
                match = re.search(r"\b(19\d{2}|20\d{2})\b", date_txt)
                if match:
                    ref_year = match.group(1)

            # DOI
            for idno in _find_all_local(bibl, "idno"):
                if str(idno.attrib.get("type", "")).lower() == "doi":
                    ref_doi = _get_text(idno)
                    break

            references.append({
                "title": ref_title,
                "authors": list(dict.fromkeys(ref_authors)),
                "journal": ref_journal,
                "year": ref_year,
                "doi": ref_doi,
            })

        return references

    def _extract_via_pymupdf(
        self, pdf_bytes: bytes, pdf_input: Any = None
    ) -> dict[str, Any]:
        """Fallback extraction using PyMuPDF fitz with heuristic font-size grouping."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if doc.page_count == 0:
            raise ValueError("PDF document has no pages.")

        all_lines: list[dict[str, Any]] = []
        full_text_list: list[str] = []

        for page_idx in range(doc.page_count):
            page = doc[page_idx]
            page_text = page.get_text("text")
            full_text_list.append(page_text)

            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if block.get("type") == 0:  # text block
                    for line in block.get("lines", []):
                        spans = line.get("spans", [])
                        line_text = "".join(s.get("text", "") for s in spans).strip()
                        if not line_text:
                            continue

                        max_size = max(s.get("size", 10.0) for s in spans)
                        is_bold = any(
                            bool(s.get("flags", 0) & 2 or "bold" in s.get("font", "").lower())
                            for s in spans
                        )
                        all_lines.append({
                            "page": page_idx,
                            "text": line_text,
                            "size": round(max_size, 1),
                            "bold": is_bold,
                        })

        full_text = "\n".join(full_text_list)

        # 1. Determine body font size
        if all_lines:
            sizes = [item["size"] for item in all_lines]
            size_counts = collections.Counter(sizes)
            body_font_size = size_counts.most_common(1)[0][0]
        else:
            body_font_size = 10.0

        # 2. Extract Metadata via PyMuPDF doc properties & regex
        doc_meta = doc.metadata or {}
        title = doc_meta.get("title")
        if not title or title.lower().endswith(".pdf") or title == "Untitled":
            # Search for largest font text on page 0
            page0_lines = [item for item in all_lines if item["page"] == 0]
            if page0_lines:
                title_line = max(page0_lines, key=lambda x: x["size"])
                title = title_line["text"]

        author_str = doc_meta.get("author")
        authors: list[str] = []
        if author_str:
            authors = [
                a.strip()
                for a in re.split(r"[,;]|\band\b", author_str)
                if a.strip()
            ]

        doi_match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", full_text)
        doi = doi_match.group(0).rstrip(".") if doi_match else None

        pmid_match = re.search(r"\bPMID:\s*(\d+)\b", full_text, re.IGNORECASE)
        pmid = pmid_match.group(1) if pmid_match else None

        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", full_text)
        year = year_match.group(1) if year_match else None

        metadata = {
            "title": title,
            "authors": authors,
            "doi": doi,
            "pmid": pmid,
            "pmcid": None,
            "journal": None,
            "publication_date": year,
            "year": year,
        }

        # 3. Group lines into chunks by font-size heuristics
        chunks: list[dict[str, str]] = []
        references: list[dict[str, Any]] = []

        section_regex = re.compile(
            r"^(?:abstract|introduction|materials?\s+and\s+methods?|methods?|results?|discussion|conclusion|references|bibliography)$",
            re.IGNORECASE,
        )

        current_section = "Body"
        current_para: list[str] = []

        def flush_paragraph():
            nonlocal current_para, current_section
            if current_para:
                para_text = " ".join(current_para).strip()
                if para_text:
                    if current_section.lower() in ("references", "bibliography"):
                        ref_doi = re.search(
                            r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", para_text
                        )
                        ref_yr = re.search(r"\b(19\d{2}|20\d{2})\b", para_text)
                        references.append({
                            "title": para_text,
                            "authors": [],
                            "journal": None,
                            "year": ref_yr.group(1) if ref_yr else None,
                            "doi": ref_doi.group(0).rstrip(".") if ref_doi else None,
                        })
                    else:
                        chunks.append({"section": current_section, "text": para_text})
                current_para = []

        for line_item in all_lines:
            txt = line_item["text"]
            size = line_item["size"]
            bold = line_item["bold"]

            is_header = False
            if section_regex.match(txt):
                is_header = True
            elif len(txt) <= 80 and (
                size >= body_font_size * 1.15 or (bold and size >= body_font_size * 1.05)
            ):
                if not txt.endswith("."):
                    is_header = True

            if is_header:
                flush_paragraph()
                current_section = txt
            else:
                current_para.append(txt)

        flush_paragraph()

        doc.close()

        return {
            "metadata": metadata,
            "chunks": chunks,
            "text_chunks": chunks,
            "references": references,
        }
