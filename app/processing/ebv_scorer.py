"""EBV Literature Affinity Scoring (ELAS) Engine.

Scores and bifurcates scientific literature based on EBV relevance:
- Tier 1 (Score >= 0.70): High EBV Affinity (direct EBV viral/host biology)
- Tier 2 (0.30 <= Score < 0.70): Moderate / Co-Infection / General Oncology
- Tier 3 (Score < 0.30): Low / Reagent Noise / Incidental mentions
"""

import re
from typing import Dict, Any

# Primary EBV Viral Genes, Antigens, and Domain Terms
EBV_CORE_TERMS = {
    "epstein-barr", "ebv", "ebna1", "ebna2", "ebna3", "ebna3a", "ebna3b", "ebna3c", "ebna-lp",
    "lmp1", "lmp2", "lmp2a", "lmp2b", "eber", "eber1", "eber2", "bzlf1", "brlf1", "zebra", "rta",
    "gp350", "gp220", "gh/gl", "gp42", "orip", "bart", "lncbart", "rpms1", "bflf1", "bglf4",
    "burkitt", "nasopharyngeal carcinoma", "ptld", "post-transplant lymphoproliferative"
}

# Negative Reagent / Non-EBV Assay Noise Terms
REAGENT_NOISE_TERMS = {
    "culture supernatant", "immortalized by ebv", "ebv-transformed lcl supernatant",
    "chagas", "trypanosoma", "hiv-1 screening", "serological control"
}

class EBVALiteratureScorer:
    """Calculates EBV Literature Affinity Score (ELAS) for a paper."""

    def __init__(self, title_weight: float = 0.40, abstract_weight: float = 0.35, intro_weight: float = 0.25):
        self.w_title = title_weight
        self.w_abstract = abstract_weight
        self.w_intro = intro_weight

    def score_paper(self, title: str = "", abstract: str = "", introduction: str = "") -> Dict[str, Any]:
        """Calculates ELAS score between 0.0 and 1.0."""
        title_lower = (title or "").lower()
        abstract_lower = (abstract or "").lower()
        intro_lower = (introduction or "").lower()
        full_text = f"{title_lower} {abstract_lower} {intro_lower}"

        # Check for heavy reagent noise terms
        noise_penalty = 0.0
        for noise in REAGENT_NOISE_TERMS:
            if noise in full_text:
                noise_penalty += 0.35

        # Section-weighted term hits
        title_hits = sum(1 for term in EBV_CORE_TERMS if term in title_lower)
        abstract_hits = sum(1 for term in EBV_CORE_TERMS if term in abstract_lower)
        intro_hits = sum(1 for term in EBV_CORE_TERMS if term in intro_lower)

        s_title = min(1.0, title_hits * 0.50)
        s_abstract = min(1.0, abstract_hits * 0.25)
        s_intro = min(1.0, intro_hits * 0.20)

        raw_score = (self.w_title * s_title) + (self.w_abstract * s_abstract) + (self.w_intro * s_intro)
        final_score = max(0.0, round(raw_score - noise_penalty, 3))

        if final_score >= 0.70:
            tier = "TIER_1_HIGH_AFFINITY"
        elif final_score >= 0.30:
            tier = "TIER_2_MODERATE"
        else:
            tier = "TIER_3_REAGENT_NOISE"

        return {
            "elas_score": final_score,
            "tier": tier,
            "title_hits": title_hits,
            "abstract_hits": abstract_hits,
            "is_relevant": final_score >= 0.30
        }
