"""NCBI GEO and SRA Metadata Crawler.

Retrieves single-cell dataset series metadata from NCBI GEO (db='gds'),
parses series matrix headers and sample characteristics (cell types, disease state,
tissue source), and stages structured JSON output in data/staging/geo/.
"""

import json
from pathlib import Path
from typing import Any

import requests


def parse_sample_characteristics(characteristics: list[str]) -> dict[str, Any]:
    """Parse sample characteristic strings into structured metadata.

    Args:
        characteristics: List of raw characteristic strings (e.g. "cell type: B cell").

    Returns:
        Dict containing cell_type, disease_state, tissue_source,
        attributes, and raw_characteristics.
    """
    cell_type: str | None = None
    disease_state: str | None = None
    tissue_source: str | None = None
    attributes: dict[str, str] = {}

    cell_type_keys = {
        "cell type",
        "cell_type",
        "cell type label",
        "cell line",
        "cell_line",
        "cell population",
        "cell subtype",
        "celltype",
    }
    disease_keys = {
        "disease",
        "disease state",
        "disease_state",
        "disease status",
        "diagnosis",
        "condition",
        "pathology",
    }
    tissue_keys = {
        "tissue",
        "tissue source",
        "tissue_source",
        "organ",
        "tissue type",
        "source name",
        "anatomical site",
    }

    for idx, char_str in enumerate(characteristics):
        char_str_clean = char_str.strip()
        if not char_str_clean:
            continue

        if ":" in char_str_clean:
            key, val = char_str_clean.split(":", 1)
            key = key.strip().strip('"').strip("'")
            val = val.strip().strip('"').strip("'")
            attributes[key] = val
            key_lower = key.lower()

            if cell_type is None and key_lower in cell_type_keys:
                cell_type = val
            elif disease_state is None and key_lower in disease_keys:
                disease_state = val
            elif tissue_source is None and key_lower in tissue_keys:
                tissue_source = val
        else:
            clean_val = char_str_clean.strip('"').strip("'")
            attributes[f"characteristic_{idx}"] = clean_val

    return {
        "cell_type": cell_type,
        "disease_state": disease_state,
        "tissue_source": tissue_source,
        "attributes": attributes,
        "raw_characteristics": characteristics,
    }


def parse_series_matrix_header(matrix_text: str) -> dict[str, Any]:
    """Parse header metadata and sample attributes from GEO Series Matrix text.

    Args:
        matrix_text: Full text of GEO series matrix or SOFT file header.

    Returns:
        Dict containing title, summary, overall_design, gds_type, organism,
        and samples list.
    """
    title_parts: list[str] = []
    summary_parts: list[str] = []
    design_parts: list[str] = []
    gds_type: str = ""
    organism: str = ""

    sample_accessions: list[str] = []
    sample_titles: list[str] = []
    sample_char_rows: list[list[str]] = []
    sample_source_names: list[str] = []
    sample_organisms: list[str] = []

    for line in matrix_text.splitlines():
        line_str = line.strip()
        if not line_str or line_str.startswith("!series_matrix_table_begin"):
            if line_str.startswith("!series_matrix_table_begin"):
                break
            continue

        if not line_str.startswith("!") and not line_str.startswith("^"):
            continue

        if line_str.startswith("!Series_title"):
            val = line_str.split("=", 1)[1].strip().strip('"')
            title_parts.append(val)
        elif line_str.startswith("!Series_summary"):
            val = line_str.split("=", 1)[1].strip().strip('"')
            summary_parts.append(val)
        elif line_str.startswith("!Series_overall_design"):
            val = line_str.split("=", 1)[1].strip().strip('"')
            design_parts.append(val)
        elif line_str.startswith("!Series_type"):
            gds_type = line_str.split("=", 1)[1].strip().strip('"')
        elif line_str.startswith("!Series_sample_organism"):
            organism = line_str.split("=", 1)[1].strip().strip('"')
        elif line_str.startswith("!Sample_geo_accession"):
            parts = line_str.split("=", 1)[1].split("\t")
            sample_accessions = [p.strip().strip('"') for p in parts if p.strip()]
        elif line_str.startswith("!Sample_title"):
            parts = line_str.split("=", 1)[1].split("\t")
            sample_titles = [p.strip().strip('"') for p in parts]
        elif line_str.startswith("!Sample_characteristics_ch1"):
            parts = line_str.split("=", 1)[1].split("\t")
            sample_char_rows.append([p.strip().strip('"') for p in parts])
        elif line_str.startswith("!Sample_source_name_ch1"):
            parts = line_str.split("=", 1)[1].split("\t")
            sample_source_names = [p.strip().strip('"') for p in parts]
        elif line_str.startswith("!Sample_organism_ch1"):
            parts = line_str.split("=", 1)[1].split("\t")
            sample_organisms = [p.strip().strip('"') for p in parts]

    samples: list[dict[str, Any]] = []
    num_samples = len(sample_accessions)

    for i in range(num_samples):
        acc = sample_accessions[i]
        sample_title = sample_titles[i] if i < len(sample_titles) else ""
        raw_chars = [row[i] for row in sample_char_rows if i < len(row) and row[i]]
        parsed_char = parse_sample_characteristics(raw_chars)

        tissue = parsed_char["tissue_source"]
        if not tissue and i < len(sample_source_names) and sample_source_names[i]:
            tissue = sample_source_names[i]

        sample_obj = {
            "accession": acc,
            "title": sample_title,
            "cell_type": parsed_char["cell_type"],
            "disease_state": parsed_char["disease_state"],
            "tissue_source": tissue,
            "attributes": parsed_char["attributes"],
            "raw_characteristics": raw_chars,
        }
        if i < len(sample_organisms) and sample_organisms[i]:
            sample_obj["organism"] = sample_organisms[i]

        samples.append(sample_obj)

    return {
        "title": " ".join(title_parts),
        "summary": " ".join(summary_parts),
        "overall_design": " ".join(design_parts),
        "gds_type": gds_type,
        "organism": organism,
        "samples": samples,
    }


class GEOCrawler:
    """Crawler for fetching and staging single-cell GEO/SRA dataset metadata."""

    def __init__(
        self,
        output_dir: str | Path = "data/staging/geo",
        base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        geo_base_url: str = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi",
        timeout: int = 30,
        session: requests.Session | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.base_url = base_url.rstrip("/")
        self.geo_base_url = geo_base_url
        self.timeout = timeout
        self.session = session or requests.Session()

    def search_gse(self, gse_id: str) -> str:
        """Search NCBI Entrez GDS database for a GSE accession ID to get its UID.

        Args:
            gse_id: GEO Series accession (e.g. 'GSE189141').

        Returns:
            Entrez UID string.

        Raises:
            ValueError: If GSE ID format is invalid or no record is found.
            RuntimeError: If Entrez API request fails.
        """
        gse_clean = gse_id.strip().upper()
        if not gse_clean.startswith("GSE") or not gse_clean[3:].isdigit():
            raise ValueError(f"Invalid GSE series ID format: '{gse_id}'")

        url = f"{self.base_url}/esearch.fcgi"
        params = {
            "db": "gds",
            "term": f"{gse_clean}[Accession]",
            "retmode": "json",
        }

        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(
                f"Entrez esearch request failed for '{gse_clean}': {e}"
            ) from e
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Invalid JSON response from esearch for '{gse_clean}': {e}"
            ) from e

        id_list = data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            raise ValueError(f"No NCBI GDS dataset found for accession '{gse_clean}'")

        return str(id_list[0])

    def fetch_summary(self, uid: str) -> dict[str, Any]:
        """Fetch summary metadata from NCBI Entrez GDS for a given UID.

        Args:
            uid: Entrez GDS UID string.

        Returns:
            Dict containing raw summary response data.

        Raises:
            ValueError: If summary record for UID is not found.
            RuntimeError: If Entrez API request fails.
        """
        url = f"{self.base_url}/esummary.fcgi"
        params = {
            "db": "gds",
            "id": uid,
            "retmode": "json",
        }

        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(
                f"Entrez esummary request failed for UID '{uid}': {e}"
            ) from e
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Invalid JSON response from esummary for UID '{uid}': {e}"
            ) from e

        result = data.get("result", {})
        if uid not in result:
            raise ValueError(
                f"Summary record for UID '{uid}' not found in Entrez response"
            )

        return result[uid]

    def fetch_series_matrix_header(self, gse_id: str) -> str:
        """Fetch the series matrix header text directly from NCBI GEO.

        Args:
            gse_id: GEO Series accession (e.g. 'GSE189141').

        Returns:
            Header text string.

        Raises:
            RuntimeError: If network request fails.
        """
        gse_clean = gse_id.strip().upper()
        url = f"{self.geo_base_url}?acc={gse_clean}&targ=self&form=text&view=brief"

        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            raise RuntimeError(
                f"Failed to fetch series matrix header for '{gse_clean}': {e}"
            ) from e

    def fetch_gse(
        self, gse_id: str, series_matrix_text: str | None = None
    ) -> dict[str, Any]:
        """Retrieve complete GSE series metadata and sample characteristics.

        Args:
            gse_id: GEO Series accession (e.g. 'GSE189141').
            series_matrix_text: Optional series matrix text content.

        Returns:
            Dict containing formatted series metadata and parsed sample list.
        """
        gse_clean = gse_id.strip().upper()
        uid = self.search_gse(gse_clean)
        summary_raw = self.fetch_summary(uid)

        if series_matrix_text is None:
            try:
                series_matrix_text = self.fetch_series_matrix_header(gse_clean)
            except Exception:
                series_matrix_text = ""

        parsed_matrix = (
            parse_series_matrix_header(series_matrix_text) if series_matrix_text else {}
        )

        title = parsed_matrix.get("title") or summary_raw.get("title", "")
        summary = parsed_matrix.get("summary") or summary_raw.get("summary", "")
        overall_design = parsed_matrix.get("overall_design") or summary_raw.get(
            "overall_design", ""
        )
        gds_type = parsed_matrix.get("gds_type") or summary_raw.get("gdsType", "")
        organism = parsed_matrix.get("organism") or summary_raw.get("organism", "")

        samples = parsed_matrix.get("samples", [])
        if not samples and "samples" in summary_raw:
            for s in summary_raw.get("samples", []):
                acc = s.get("accession", "")
                stitle = s.get("title", "")
                samples.append(
                    {
                        "accession": acc,
                        "title": stitle,
                        "cell_type": None,
                        "disease_state": None,
                        "tissue_source": None,
                        "attributes": {},
                        "raw_characteristics": [],
                    }
                )

        return {
            "gse_id": gse_clean,
            "uid": uid,
            "title": title,
            "summary": summary,
            "overall_design": overall_design,
            "gds_type": gds_type,
            "organism": organism,
            "samples": samples,
        }

    def save_staging(
        self, gse_data: dict[str, Any], output_path: str | Path | None = None
    ) -> Path:
        """Save structured GSE dataset metadata into JSON staging directory.

        Args:
            gse_data: Structured GSE metadata dict.
            output_path: Optional custom output file path.

        Returns:
            Path object pointing to the written JSON file.
        """
        if not output_path:
            gse_id = gse_data.get("gse_id", "GSE_UNKNOWN")
            output_path = self.output_dir / f"{gse_id}.json"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(gse_data, f, indent=2, ensure_ascii=False)

        return output_path

    def crawl_and_stage(
        self, gse_id: str, series_matrix_text: str | None = None
    ) -> Path:
        """Convenience method to fetch GSE metadata and save to staging.

        Args:
            gse_id: GEO Series accession (e.g. 'GSE189141').
            series_matrix_text: Optional series matrix text content.

        Returns:
            Path of written JSON file in data/staging/geo/.
        """
        data = self.fetch_gse(gse_id, series_matrix_text=series_matrix_text)
        return self.save_staging(data)
