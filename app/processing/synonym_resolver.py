"""Local dictionary-based synonym resolver with EMBL-EBI OLS fallback."""

import csv
import json
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Standard categories and their ontology prefixes
CATEGORY_MAP: dict[str, str] = {
    "GENE": "hgnc",
    "HGNC": "hgnc",
    "CELL_TYPE": "cl",
    "CELL": "cl",
    "CL": "cl",
    "DISEASE": "doid",
    "DOID": "doid",
    "PROTEIN": "uniprot",
    "UNIPROT": "uniprot",
    "ANATOMY": "uberon",
    "UBERON": "uberon",
}

OLS_API_URL = "https://www.ebi.ac.uk/ols4/api/search"


def normalize_term(term: str) -> str:
    """Normalize input term string for token matching."""
    if not term:
        return ""
    text = term.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class SynonymResolver:
    """Resolves entity terms using local dictionaries and OLS fallback."""

    def __init__(
        self,
        dictionary_dir: str | Path | None = None,
        dictionaries: dict[str, Any] | None = None,
        ols_enabled: bool = True,
        ols_timeout: float = 5.0,
        fuzzy_threshold: float = 0.8,
    ):
        """Initialize SynonymResolver."""
        self.ols_enabled = ols_enabled
        self.ols_timeout = ols_timeout
        self.fuzzy_threshold = fuzzy_threshold

        # Storage structure per category:
        # _raw_map[category][raw_lowercased_term] -> record
        # _norm_map[category][normalized_term] -> record
        # _term_list[category] -> [(normalized_term, record), ...]
        self._raw_map: dict[str, dict[str, dict[str, Any]]] = {}
        self._norm_map: dict[str, dict[str, dict[str, Any]]] = {}
        self._term_list: dict[str, list[tuple[str, dict[str, Any]]]] = {}

        if dictionary_dir:
            self.load_from_directory(dictionary_dir)

        if dictionaries:
            for category, data in dictionaries.items():
                self.load_dictionary(category, data)

    def _get_category_key(self, category: str) -> str:
        """Map input category string to canonical key."""
        if not category or not category.strip():
            raise ValueError("Category must be a non-empty string.")
        cat_clean = category.strip()
        cat_upper = cat_clean.upper()
        if cat_upper in CATEGORY_MAP:
            return CATEGORY_MAP[cat_upper]
        cat_lower = cat_clean.lower()
        if cat_lower in ("hgnc", "cl", "doid", "uniprot", "uberon"):
            return cat_lower
        raise ValueError(
            f"Unsupported category: '{category}'. "
            "Supported: HGNC/GENE, CL/CELL_TYPE, DOID/DISEASE, UNIPROT, UBERON."
        )

    def load_dictionary(
        self, category: str, source: str | Path | dict[str, Any] | list[Any]
    ) -> None:
        """Load dictionary entries for a specified category."""
        cat_key = self._get_category_key(category)

        if cat_key not in self._raw_map:
            self._raw_map[cat_key] = {}
            self._norm_map[cat_key] = {}
            self._term_list[cat_key] = []

        entries = self._parse_source(source)
        for entry in entries:
            term = entry.get("term")
            canonical_id = entry.get("canonical_id") or entry.get("id")
            symbol = entry.get("symbol") or entry.get("name") or term
            aliases = entry.get("aliases") or entry.get("synonyms") or []

            if not term or not canonical_id:
                continue

            record = {
                "canonical_id": canonical_id,
                "symbol": symbol,
                "category": cat_key,
                "aliases": aliases,
            }

            all_terms = [term] + (
                list(aliases) if isinstance(aliases, (list, tuple)) else [aliases]
            )
            for t in all_terms:
                if not t or not isinstance(t, str):
                    continue
                raw_t = t.strip().lower()
                norm_t = normalize_term(t)

                if raw_t:
                    self._raw_map[cat_key][raw_t] = record
                if norm_t:
                    self._norm_map[cat_key][norm_t] = record
                    self._term_list[cat_key].append((norm_t, record))

    def _parse_source(
        self, source: str | Path | dict[str, Any] | list[Any]
    ) -> list[dict[str, Any]]:
        """Parse source data into a list of entry dicts."""
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Dictionary file not found: {path}")
            if path.suffix.lower() == ".csv":
                return self._parse_csv(path)
            elif path.suffix.lower() == ".json":
                return self._parse_json(path)
            else:
                raise ValueError(
                    f"Unsupported file format: {path.suffix}. Expected CSV or JSON."
                )
        elif isinstance(source, dict):
            entries = []
            for term, val in source.items():
                if isinstance(val, str):
                    entries.append({"term": term, "canonical_id": val})
                elif isinstance(val, dict):
                    rec = dict(val)
                    if "term" not in rec:
                        rec["term"] = term
                    entries.append(rec)
            return entries
        elif isinstance(source, list):
            return source
        else:
            raise ValueError(f"Invalid dictionary source type: {type(source)}")

    def _parse_csv(self, path: Path) -> list[dict[str, Any]]:
        entries = []
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return entries

            header_lower = [h.strip().lower() for h in header]
            term_idx = -1
            id_idx = -1
            symbol_idx = -1
            aliases_idx = -1

            for idx, h in enumerate(header_lower):
                if h in ("term", "synonym", "alias", "name"):
                    if term_idx == -1:
                        term_idx = idx
                elif h in ("id", "canonical_id", "obo_id", "accession"):
                    if id_idx == -1:
                        id_idx = idx
                elif h in ("symbol", "preferred_label", "label"):
                    symbol_idx = idx
                elif h in ("aliases", "synonyms"):
                    aliases_idx = idx

            if term_idx == -1 and len(header) >= 1:
                term_idx = 0
            if id_idx == -1 and len(header) >= 2:
                id_idx = 1

            for row in reader:
                if not row or len(row) <= max(term_idx, id_idx):
                    continue
                term = row[term_idx].strip()
                cid = row[id_idx].strip()
                sym = (
                    row[symbol_idx].strip()
                    if symbol_idx != -1 and len(row) > symbol_idx
                    else term
                )
                aliases = []
                if aliases_idx != -1 and len(row) > aliases_idx and row[aliases_idx]:
                    aliases = [
                        a.strip()
                        for a in re.split(r"[;|]", row[aliases_idx])
                        if a.strip()
                    ]

                if term and cid:
                    entries.append(
                        {
                            "term": term,
                            "canonical_id": cid,
                            "symbol": sym,
                            "aliases": aliases,
                        }
                    )
        return entries

    def _parse_json(self, path: Path) -> list[dict[str, Any]]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return self._parse_source(data)

    def load_from_directory(self, directory_dir: str | Path) -> None:
        """Load dictionary files matching category names from a directory."""
        dir_path = Path(directory_dir)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Dictionary directory not found: {dir_path}")

        for file_path in dir_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in (".csv", ".json"):
                stem_upper = file_path.stem.upper()
                if stem_upper in CATEGORY_MAP or file_path.stem.lower() in (
                    "hgnc",
                    "cl",
                    "doid",
                    "uniprot",
                    "uberon",
                ):
                    self.load_dictionary(file_path.stem, file_path)

    def resolve(self, term: str, category: str | None = None) -> dict[str, Any] | None:
        """Resolve an extracted entity term to a canonical ID record.

        Args:
            term: Entity text string to resolve.
            category: Optional category filter (e.g. HGNC, CELL_TYPE, DISEASE).

        Returns:
            Dict with canonical_id, symbol, category, confidence, match_type, source,
            or None if no match found.
        """
        if not term or not term.strip():
            return None

        clean_term = term.strip()

        if category:
            cat_keys = [self._get_category_key(category)]
        else:
            cat_keys = list(self._raw_map.keys())

        # 1. Exact raw match
        raw_t = clean_term.lower()
        for cat in cat_keys:
            if cat in self._raw_map and raw_t in self._raw_map[cat]:
                record = self._raw_map[cat][raw_t]
                return {
                    "canonical_id": record["canonical_id"],
                    "symbol": record["symbol"],
                    "category": record["category"],
                    "confidence": 1.0,
                    "match_type": "exact",
                    "source": f"local_{cat}",
                }

        # 2. Exact normalized match
        norm_t = normalize_term(clean_term)
        if norm_t:
            for cat in cat_keys:
                if cat in self._norm_map and norm_t in self._norm_map[cat]:
                    record = self._norm_map[cat][norm_t]
                    return {
                        "canonical_id": record["canonical_id"],
                        "symbol": record["symbol"],
                        "category": record["category"],
                        "confidence": 0.98,
                        "match_type": "exact",
                        "source": f"local_{cat}",
                    }

        # 3. Fuzzy match
        if norm_t:
            best_match = None
            best_score = 0.0
            best_cat = None

            for cat in cat_keys:
                if cat not in self._term_list:
                    continue
                for dict_norm_t, record in self._term_list[cat]:
                    score = SequenceMatcher(None, norm_t, dict_norm_t).ratio()
                    if score > best_score:
                        best_score = score
                        best_match = record
                        best_cat = cat

            if best_match and best_score >= self.fuzzy_threshold:
                return {
                    "canonical_id": best_match["canonical_id"],
                    "symbol": best_match["symbol"],
                    "category": best_match["category"],
                    "confidence": round(best_score, 3),
                    "match_type": "fuzzy",
                    "source": f"local_{best_cat}",
                }

        # 4. Fallback query to EMBL-EBI OLS API
        if self.ols_enabled:
            return self._resolve_ols(clean_term, category)

        return None

    def _resolve_ols(
        self, term: str, category: str | None = None
    ) -> dict[str, Any] | None:
        """Query EMBL-EBI OLS API for term resolution as fallback."""
        ontology_filter = None
        cat_key = None
        if category:
            cat_key = self._get_category_key(category)
            ontology_filter = cat_key

        params = {"q": term, "rows": 5}
        if ontology_filter:
            params["ontology"] = ontology_filter

        try:
            response = requests.get(
                OLS_API_URL, params=params, timeout=self.ols_timeout
            )
            if response.status_code != 200:
                logger.warning(f"OLS API returned status code {response.status_code}")
                return None

            data = response.json()
            docs = data.get("response", {}).get("docs", [])
            if not docs:
                return None

            top_doc = docs[0]
            canonical_id = (
                top_doc.get("obo_id") or top_doc.get("short_form") or top_doc.get("id")
            )
            symbol = top_doc.get("label") or term
            ont_prefix = top_doc.get("ontology_prefix") or ontology_filter or "ols"

            if not canonical_id:
                return None

            doc_label = top_doc.get("label", "").lower()
            if doc_label == term.lower():
                confidence = 0.95
            elif normalize_term(doc_label) == normalize_term(term):
                confidence = 0.90
            else:
                confidence = 0.85

            return {
                "canonical_id": canonical_id,
                "symbol": symbol,
                "category": cat_key or ont_prefix.lower(),
                "confidence": confidence,
                "match_type": "ols",
                "source": f"ols_{ont_prefix.lower()}",
            }
        except requests.RequestException as e:
            logger.warning(f"OLS API request failed for '{term}': {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing OLS API response for '{term}': {e}")
            return None
