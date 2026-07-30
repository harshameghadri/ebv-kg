"""PMC JATS XML Parser.

Extracts article metadata, hierarchical paragraph text chunks, and bibliography references
from standard PMC JATS XML documents.
"""

import os
from pathlib import Path
from typing import Any

try:
    from lxml import etree
except ImportError:
    etree = None

import xml.etree.ElementTree as ET


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


class PMCXMLParser:
    """Parser class for JATS PMC XML files."""

    def parse(self, xml_input: str | bytes | Path) -> dict[str, Any]:
        """Parse PMC JATS XML from string, bytes, or file path.

        Args:
            xml_input: XML string, raw bytes, or Path object to XML file.

        Returns:
            Dict containing 'metadata', 'text_chunks', and 'references'.
        """
        if isinstance(xml_input, Path):
            if not xml_input.exists():
                raise FileNotFoundError(f"File not found: {xml_input}")
            content = xml_input.read_bytes()
        elif isinstance(xml_input, str):
            if os.path.isfile(xml_input):
                content = Path(xml_input).read_bytes()
            else:
                content = xml_input.encode("utf-8")
        elif isinstance(xml_input, bytes):
            content = xml_input
        else:
            raise TypeError(f"Unsupported xml_input type: {type(xml_input)}")

        if not content.strip():
            raise ValueError("Empty XML content provided.")

        root = _parse_xml_bytes(content)

        # Handle wrapper tags like <pmc-articleset>
        if _local_name(root.tag) != "article":
            article_elem = _find_descendant_local(root, "article")
            if article_elem is not None:
                root = article_elem
            else:
                raise ValueError("PMC XML content does not contain an <article> element.")

        return {
            "metadata": self.extract_metadata(root),
            "text_chunks": self.extract_text_chunks(root),
            "references": self.extract_references(root),
        }

    def extract_metadata(self, root: Any) -> dict[str, Any]:
        """Extract article metadata (Title, Journal, DOI, PMID, PMCID, Pub Date, Authors)."""
        front = _find_descendant_local(root, "front")
        if front is None:
            front = root

        article_meta = _find_descendant_local(front, "article-meta")
        journal_meta = _find_descendant_local(front, "journal-meta")

        # Article Title
        title: str | None = None
        if article_meta is not None:
            title_group = _find_descendant_local(article_meta, "title-group")
            if title_group is not None:
                title_elem = _find_child_local(title_group, "article-title")
                if title_elem is not None:
                    title = _get_text(title_elem)
        if not title:
            title_elem = _find_descendant_local(front, "article-title")
            if title_elem is not None:
                title = _get_text(title_elem)

        # Journal Name
        journal: str | None = None
        if journal_meta is not None:
            journal_title_group = _find_descendant_local(
                journal_meta, "journal-title-group"
            )
            if journal_title_group is not None:
                j_title_elem = _find_child_local(journal_title_group, "journal-title")
                if j_title_elem is not None:
                    journal = _get_text(j_title_elem)
            if not journal:
                j_title_elem = _find_descendant_local(journal_meta, "journal-title")
                if j_title_elem is not None:
                    journal = _get_text(j_title_elem)
            if not journal:
                j_id_elem = _find_descendant_local(journal_meta, "journal-id")
                if j_id_elem is not None:
                    journal = _get_text(j_id_elem)

        # DOI, PMID, PMCID
        doi: str | None = None
        pmid: str | None = None
        pmcid: str | None = None
        if article_meta is not None:
            for elem in _find_all_local(article_meta, "article-id"):
                id_type = str(elem.attrib.get("pub-id-type", "")).lower()
                val = _get_text(elem)
                if val:
                    if id_type == "doi":
                        doi = val
                    elif id_type == "pmid":
                        pmid = val
                    elif id_type in ("pmc", "pmcid"):
                        pmcid = val

        # Publication Date
        pub_date = self._extract_pub_date(article_meta)

        # Authors
        authors = self._extract_authors(article_meta)

        return {
            "title": title,
            "journal": journal,
            "doi": doi,
            "pmid": pmid,
            "pmcid": pmcid,
            "publication_date": pub_date,
            "authors": authors,
        }

    def _extract_pub_date(self, article_meta: Any | None) -> str | None:
        """Extract and format publication date string (YYYY-MM-DD, YYYY-MM, or YYYY)."""
        if article_meta is None:
            return None
        pub_date_elems = _find_all_local(article_meta, "pub-date")
        if not pub_date_elems:
            return None

        # Prioritize epub -> ppub -> pmc-release -> collection -> first available
        preferred_types = ["epub", "ppub", "pmc-release", "collection"]
        selected_elem = pub_date_elems[0]
        for p_type in preferred_types:
            found = False
            for elem in pub_date_elems:
                if str(elem.attrib.get("pub-type", "")).lower() == p_type:
                    selected_elem = elem
                    found = True
                    break
            if found:
                break

        year_elem = _find_child_local(selected_elem, "year")
        month_elem = _find_child_local(selected_elem, "month")
        day_elem = _find_child_local(selected_elem, "day")

        year = _get_text(year_elem) if year_elem is not None else ""
        month = _get_text(month_elem) if month_elem is not None else ""
        day = _get_text(day_elem) if day_elem is not None else ""

        if not year:
            return None

        if month:
            if month.isdigit():
                month = f"{int(month):02d}"
            if day:
                if day.isdigit():
                    day = f"{int(day):02d}"
                return f"{year}-{month}-{day}"
            return f"{year}-{month}"
        return year

    def _extract_authors(self, article_meta: Any | None) -> list[str]:
        """Extract list of author names formatted as 'Given Surname'."""
        if article_meta is None:
            return []

        authors: list[str] = []
        contrib_groups = _find_all_local(article_meta, "contrib-group")
        for cg in contrib_groups:
            contribs = _find_all_local(cg, "contrib")
            for contrib in contribs:
                contrib_type = str(contrib.attrib.get("contrib-type", "")).lower()
                if contrib_type and contrib_type not in ("author", "autor"):
                    continue

                name_elem = _find_child_local(contrib, "name")
                if name_elem is None:
                    name_elem = _find_descendant_local(contrib, "name")
                if name_elem is not None:
                    surname_elem = _find_child_local(name_elem, "surname")
                    given_elem = _find_child_local(name_elem, "given-names")
                    surname = _get_text(surname_elem) if surname_elem is not None else ""
                    given = _get_text(given_elem) if given_elem is not None else ""

                    if given and surname:
                        authors.append(f"{given} {surname}")
                    elif surname:
                        authors.append(surname)
                    elif given:
                        authors.append(given)
                else:
                    collab_elem = _find_child_local(contrib, "collab")
                    if collab_elem is not None:
                        collab_text = _get_text(collab_elem)
                        if collab_text:
                            authors.append(collab_text)

        # Fallback if no authors extracted from contrib-group
        if not authors:
            for name_elem in _find_all_local(article_meta, "name"):
                surname_elem = _find_child_local(name_elem, "surname")
                given_elem = _find_child_local(name_elem, "given-names")
                surname = _get_text(surname_elem) if surname_elem is not None else ""
                given = _get_text(given_elem) if given_elem is not None else ""
                if given and surname:
                    authors.append(f"{given} {surname}")
                elif surname:
                    authors.append(surname)

        return authors

    def extract_text_chunks(self, root: Any) -> list[dict[str, str]]:
        """Extract paragraph-by-paragraph text chunks organized by section."""
        body = _find_descendant_local(root, "body")
        if body is None:
            return []

        chunks: list[dict[str, str]] = []

        def process_container(container: Any, section_path: list[str]) -> None:
            for child in container:
                tag = _local_name(child.tag)
                if tag == "title":
                    continue
                elif tag == "p":
                    text = _get_text(child)
                    if text:
                        sec_name = " > ".join(section_path) if section_path else "Body"
                        chunks.append({"section": sec_name, "text": text})
                elif tag == "sec":
                    sec_title_elem = _find_child_local(child, "title")
                    sec_title = (
                        _get_text(sec_title_elem) if sec_title_elem is not None else ""
                    )
                    new_path = list(section_path)
                    if sec_title:
                        new_path.append(sec_title)
                    process_container(child, new_path)
                elif tag in ("boxed-text", "disp-quote", "list", "list-item"):
                    process_container(child, section_path)

        process_container(body, [])
        return chunks

    def extract_references(self, root: Any) -> list[dict[str, Any]]:
        """Extract bibliography references from <ref-list> elements."""
        refs: list[dict[str, Any]] = []
        ref_elems = _find_all_local(root, "ref")

        for ref in ref_elems:
            citation_elem = _find_descendant_local(ref, "element-citation")
            if citation_elem is None:
                citation_elem = _find_descendant_local(ref, "mixed-citation")
            if citation_elem is None:
                citation_elem = _find_descendant_local(ref, "citation")
            if citation_elem is None:
                citation_elem = ref

            # DOI
            doi: str | None = None
            for pub_id in _find_all_local(citation_elem, "pub-id"):
                if str(pub_id.attrib.get("pub-id-type", "")).lower() == "doi":
                    doi = _get_text(pub_id)
                    break

            # Title
            title: str | None = None
            title_elem = _find_child_local(citation_elem, "article-title")
            if title_elem is None:
                title_elem = _find_child_local(citation_elem, "chapter-title")
            if title_elem is None:
                title_elem = _find_descendant_local(citation_elem, "article-title")
            if title_elem is not None:
                title = _get_text(title_elem).rstrip(".")

            # Authors
            authors: list[str] = []
            person_group = _find_descendant_local(citation_elem, "person-group")
            names_source = person_group if person_group is not None else citation_elem
            for name_elem in _find_all_local(names_source, "name"):
                surname_elem = _find_child_local(name_elem, "surname")
                given_elem = _find_child_local(name_elem, "given-names")
                surname = _get_text(surname_elem) if surname_elem is not None else ""
                given = _get_text(given_elem) if given_elem is not None else ""
                if given and surname:
                    authors.append(f"{given} {surname}")
                elif surname:
                    authors.append(surname)
                elif given:
                    authors.append(given)

            if not authors:
                collab = _find_descendant_local(citation_elem, "collab")
                if collab is not None:
                    collab_text = _get_text(collab)
                    if collab_text:
                        authors.append(collab_text)

            # Journal / Source
            journal: str | None = None
            source_elem = _find_child_local(citation_elem, "source")
            if source_elem is None:
                source_elem = _find_child_local(citation_elem, "journal-title")
            if source_elem is not None:
                journal = _get_text(source_elem)

            # Year
            year: str | None = None
            year_elem = _find_child_local(citation_elem, "year")
            if year_elem is not None:
                year = _get_text(year_elem)

            refs.append({
                "doi": doi,
                "title": title,
                "authors": authors,
                "journal": journal,
                "year": year,
            })

        return refs
