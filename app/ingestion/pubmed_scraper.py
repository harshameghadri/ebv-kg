"""PubMed API Scraper.

Interacts with NCBI Entrez E-utilities API to query PubMed articles,
fetch metadata, and download PMC full-text JATS XMLs (or store metadata).
"""

import json
import os
import subprocess
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


class PubMedScraper:
    """Scraper class for NCBI PubMed / PMC E-utilities API."""

    def __init__(
        self,
        staging_dir: str | Path = "data/staging",
        email: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize PubMed scraper with staging directories and credentials.

        Args:
            staging_dir: Base directory for staging output.
            email: Optional contact email for NCBI Entrez API requests.
            api_key: Optional NCBI API key for higher rate limits.
        """
        self.staging_dir = Path(staging_dir)
        self.xml_dir = self.staging_dir / "xml"
        self.metadata_dir = self.staging_dir / "metadata"
        self.email = email or os.getenv("NCBI_EMAIL") or os.getenv("ENTREZ_EMAIL") or "harshameghadri@gmail.com"
        self.api_key = api_key or os.getenv("NCBI_KEY") or os.getenv("NCBI_API_KEY")

        self.xml_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)


    def _build_params(self, base_params: dict[str, Any]) -> dict[str, Any]:
        """Add email and api_key to request parameters if provided."""
        params = dict(base_params)
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
        """Helper to make HTTP requests with exponential backoff on 429 and network errors."""
        import time
        retries = 5
        backoff = 2.0
        for i in range(retries):
            try:
                if method.upper() == "GET":
                    response = requests.get(url, **kwargs)
                else:
                    response = requests.request(method, url, **kwargs)
                if response.status_code == 429:
                    time.sleep(backoff * (2 ** i))
                    continue
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                if i == retries - 1:
                    raise e
                time.sleep(backoff * (2 ** i))
        raise requests.RequestException("Max retries exceeded")

    def search(self, query: str, max_results: int = 20) -> list[str]:
        """Search PubMed for matching PMIDs.

        Args:
            query: Search term string.
            max_results: Maximum number of PMIDs to return.

        Returns:
            List of PMID strings matching query.
        """
        if not query or not query.strip():
            return []

        params = self._build_params(
            {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": max_results,
            }
        )

        response = self._request_with_retry("GET", ESEARCH_URL, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        result = data.get("esearchresult", {})
        id_list = result.get("idlist", [])
        return [str(pmid) for pmid in id_list]

    def fetch_metadata(self, pmids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch article metadata for PMIDs via PubMed esummary API in batches.

        Args:
            pmids: List of PMID strings.

        Returns:
            Dict mapping PMID to article metadata dict.
        """
        if not pmids:
            return {}

        batch_size = 100
        metadata_by_pmid: dict[str, dict[str, Any]] = {}

        for start_idx in range(0, len(pmids), batch_size):
            batch = pmids[start_idx : start_idx + batch_size]
            params = self._build_params(
                {
                    "db": "pubmed",
                    "id": ",".join(batch),
                    "retmode": "json",
                }
            )

            try:
                response = self._request_with_retry("GET", ESUMMARY_URL, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                logger.error("Failed to fetch summary batch starting at index %d: %s", start_idx, e)
                continue

            result_dict = data.get("result", {})

            for pmid in batch:
                if pmid not in result_dict or not isinstance(result_dict[pmid], dict):
                    continue
                item = result_dict[pmid]

                doi = None
                pmcid = None
                article_ids = item.get("articleids", [])
                for aid in article_ids:
                    if isinstance(aid, dict):
                        id_type = str(aid.get("idtype", "")).lower()
                        val = str(aid.get("value", ""))
                        if id_type == "doi":
                            doi = val
                        elif id_type in ("pmc", "pmcid"):
                            pmcid = val if val.startswith("PMC") else f"PMC{val}"

                authors = []
                for author in item.get("authors", []):
                    if isinstance(author, dict) and "name" in author:
                        authors.append(author["name"])

                pubtypes = [str(pt).lower() for pt in item.get("pubtype", [])]
                retracted = any("retract" in pt for pt in pubtypes)

                metadata_by_pmid[pmid] = {
                    "pmid": pmid,
                    "pmcid": pmcid,
                    "doi": doi,
                    "title": item.get("title", ""),
                    "journal": item.get("source", ""),
                    "publication_date": item.get("pubdate", ""),
                    "authors": authors,
                    "retracted": retracted,
                }

        return metadata_by_pmid


    def fetch_pmc_xml(self, pmcid_or_pmid: str) -> str | None:
        """Attempt to fetch full-text JATS XML from PMC efetch API.

        Args:
            pmcid_or_pmid: PMC ID (e.g. PMC10987654) or PMID string.

        Returns:
            JATS XML content string if valid full text is retrieved, else None.
        """
        if not pmcid_or_pmid:
            return None

        params = self._build_params(
            {
                "db": "pmc",
                "id": pmcid_or_pmid,
                "retmode": "xml",
            }
        )

        try:
            response = self._request_with_retry("GET", EFETCH_URL, params=params, timeout=30)
            if response.status_code != 200:
                return None

            content = response.text.strip()
            content_lower = content.lower()
            has_pmc = "<pmc-articleset" in content_lower
            has_article = "<article" in content_lower
            if not content or "<error" in content_lower or not (has_pmc or has_article):
                return None

            return content
        except requests.RequestException:
            return None

    def fetch_pubmed_abstract(self, pmid: str) -> str | None:
        """Fetch abstract text for a PMID from PubMed efetch API.

        Args:
            pmid: PMID string.

        Returns:
            Abstract text string if available, otherwise None.
        """
        if not pmid:
            return None

        params = self._build_params(
            {
                "db": "pubmed",
                "id": pmid,
                "retmode": "xml",
            }
        )

        try:
            response = self._request_with_retry("GET", EFETCH_URL, params=params, timeout=30)
            if response.status_code != 200:
                return None

            root = ET.fromstring(response.content)
            abstract_nodes = root.findall(".//AbstractText")
            if not abstract_nodes:
                return None

            parts = []
            for node in abstract_nodes:
                label = node.attrib.get("Label")
                text = "".join(node.itertext()).strip()
                if label:
                    parts.append(f"{label}: {text}")
                else:
                    parts.append(text)
            return " ".join(parts) if parts else None
        except Exception:
            return None

    def fetch_via_paperclip(self, pmid_or_pmcid: str) -> dict[str, Any] | None:
        """Fetch article metadata and full text from Paperclip CLI as a fallback."""
        import subprocess
        import json
        import shutil

        paper_id = pmid_or_pmcid
        ids_to_try = [paper_id]
        if paper_id.isdigit():
            ids_to_try.append(f"PMC{paper_id}")
        elif paper_id.startswith("PMC") and paper_id[3:].isdigit():
            ids_to_try.append(paper_id[3:])

        paperclip_bin = shutil.which("paperclip") or os.path.expanduser("~/.local/bin/paperclip")
        if not os.path.exists(paperclip_bin):
            return None

        for pid in ids_to_try:
            try:
                # 1. Fetch metadata
                cmd_meta = [paperclip_bin, "cat", f"/papers/{pid}/meta.json"]
                res_meta = subprocess.run(cmd_meta, capture_output=True, text=True, timeout=20)
                if res_meta.returncode != 0:
                    continue

                meta = json.loads(res_meta.stdout.strip())

                # 2. Fetch sections list and text chunks
                cmd_sections_ls = [paperclip_bin, "ls", f"/papers/{pid}/sections/"]
                res_ls = subprocess.run(cmd_sections_ls, capture_output=True, text=True, timeout=20)

                text_chunks = []
                if res_ls.returncode == 0:
                    sec_files = res_ls.stdout.strip().split()
                    for sec_file in sec_files:
                        if sec_file.endswith(".lines"):
                            section_name = sec_file[:-6]
                            cmd_sec_cat = [paperclip_bin, "cat", f"/papers/{pid}/sections/{sec_file}"]
                            res_sec = subprocess.run(cmd_sec_cat, capture_output=True, text=True, timeout=20)
                            if res_sec.returncode == 0:
                                lines = []
                                for line in res_sec.stdout.splitlines():
                                    line_clean = line.strip()
                                    if line_clean.startswith("L") and ":" in line_clean:
                                        parts = line_clean.split(":", 1)
                                        if parts[0][1:].isdigit():
                                            line_clean = parts[1].strip()
                                    lines.append(line_clean)
                                text_chunks.append({
                                    "section": section_name,
                                    "text": "\n".join(lines)
                                })

                if not text_chunks:
                    cmd_content = [paperclip_bin, "cat", f"/papers/{pid}/content.lines"]
                    res_content = subprocess.run(cmd_content, capture_output=True, text=True, timeout=20)
                    if res_content.returncode == 0:
                        lines = []
                        for line in res_content.stdout.splitlines():
                            line_clean = line.strip()
                            if line_clean.startswith("L") and ":" in line_clean:
                                parts = line_clean.split(":", 1)
                                if parts[0][1:].isdigit():
                                    line_clean = parts[1].strip()
                            lines.append(line_clean)
                        text_chunks.append({
                            "section": "FullText",
                            "text": "\n".join(lines)
                        })

                return {
                    "title": meta.get("title") or meta.get("document_title"),
                    "journal": meta.get("journal") or meta.get("source"),
                    "doi": meta.get("doi"),
                    "pmid": meta.get("pmid"),
                    "pmcid": meta.get("pmc_id") or meta.get("pmcid"),
                    "pub_date": meta.get("pub_date") or meta.get("published_date"),
                    "abstract": meta.get("abstract"),
                    "text_chunks": text_chunks
                }
            except Exception:
                continue

        return None

    def scrape(self, query: str, max_results: int = 20) -> dict[str, Any]:
        """Query PubMed, fetch metadata, download PMC XML or store abstract metadata.

        Args:
            query: Search query string.
            max_results: Max PMIDs to retrieve.

        Returns:
            Dict containing query summary, saved XML paths, and metadata paths.
        """
        pmids = self.search(query, max_results=max_results)
        if not pmids:
            return {
                "query": query,
                "total_found": 0,
                "xml_saved": [],
                "metadata_saved": [],
            }

        metadata_map = self.fetch_metadata(pmids)
        xml_saved = []
        metadata_saved = []

        for pmid in pmids:
            xml_path = self.xml_dir / f"{pmid}.xml"
            meta_path = self.metadata_dir / f"{pmid}.json"
            if xml_path.exists():
                xml_saved.append(str(xml_path))
                continue
            if meta_path.exists():
                metadata_saved.append(str(meta_path))
                continue

            meta = metadata_map.get(
                pmid,
                {
                    "pmid": pmid,
                    "pmcid": None,
                    "doi": None,
                    "title": "",
                    "journal": "",
                    "publication_date": "",
                    "authors": [],
                },
            )

            pmc_id = meta.get("pmcid") or pmid
            xml_content = self.fetch_pmc_xml(pmc_id)

            if xml_content:
                xml_path.write_text(xml_content, encoding="utf-8")
                xml_saved.append(str(xml_path))
            else:
                # Try paperclip fallback first
                paperclip_data = self.fetch_via_paperclip(pmc_id)
                if paperclip_data:
                    meta_path.write_text(json.dumps(paperclip_data, indent=2), encoding="utf-8")
                    metadata_saved.append(str(meta_path))
                else:
                    # Fallback to standard abstract
                    abstract = self.fetch_pubmed_abstract(pmid)
                    meta["abstract"] = abstract
                    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                    metadata_saved.append(str(meta_path))

        return {
            "query": query,
            "total_found": len(pmids),
            "xml_saved": xml_saved,
            "metadata_saved": metadata_saved,
        }


def main() -> None:
    """CLI entrypoint for PubMed scraper."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PubMed API Scraper for EBV Knowledge System"
    )
    parser.add_argument(
        "--query", "-q", required=True, help="Search query string"
    )
    parser.add_argument(
        "--max-results", "-m", type=int, default=20, help="Max results to fetch"
    )
    parser.add_argument(
        "--staging-dir",
        "-s",
        default="data/staging",
        help="Base staging directory path",
    )
    args = parser.parse_args()

    scraper = PubMedScraper(staging_dir=args.staging_dir)
    res = scraper.scrape(args.query, max_results=args.max_results)
    print(f"Scraped {res['total_found']} articles for query '{res['query']}'.")
    print(f"Saved {len(res['xml_saved'])} XML files to {scraper.xml_dir}")
    print(
        f"Saved {len(res['metadata_saved'])} metadata files to {scraper.metadata_dir}"
    )


if __name__ == "__main__":
    main()
