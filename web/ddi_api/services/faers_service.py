"""
FAERS (FDA Adverse Event Reporting System) Integration Service

Queries the openFDA FAERS API for post-market adverse event data.
For a given drug pair, retrieves reports where both drugs appear and
extracts: report counts, top adverse reactions, seriousness breakdown,
reporter types, and computes a signal score.

API Documentation: https://open.fda.gov/apis/drug/event/
Rate Limit: 240 requests per minute without API key
"""

import time
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests
from django.conf import settings
from django.core.cache import cache

logger = logging.getLogger(__name__)


@dataclass
class FaersResult:
    """Structured result from FAERS adverse event query."""
    drug_pair: Tuple[str, str]
    total_reports: int
    top_reactions: List[Dict[str, object]]      # [{'reaction': str, 'count': int}, ...]
    seriousness_stats: Dict[str, int]            # {'death': n, 'hospitalization': n, ...}
    signal_score: float                          # 0.0-1.0 composite risk signal
    reporter_breakdown: Dict[str, int]           # {'physician': n, 'consumer': n, ...}
    query_url: str = ''


class FaersService:
    """
    Queries the openFDA FAERS API for post-market adverse event data
    involving a drug pair.

    Includes rate limiting (FDA allows 240 req/min) and 24-hour caching.
    """

    # Class-level rate limiting -- FDA allows 240 requests/minute
    _last_request_time = 0
    _min_request_interval = 0.25  # 250ms = max 240 req/min

    def __init__(self):
        config = getattr(settings, 'FAERS_CONFIG', {})
        self.base_url = config.get('base_url', 'https://api.fda.gov/drug/event.json')
        self.max_results = config.get('max_results', 100)
        self.timeout = config.get('timeout_seconds', 15)
        self.cache_ttl = config.get('cache_ttl_hours', 24) * 3600
        self.max_retries = 3

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _rate_limit_wait(self):
        """Enforce rate limiting between API requests."""
        elapsed = time.time() - FaersService._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        FaersService._last_request_time = time.time()

    # ------------------------------------------------------------------
    # HTTP with retry
    # ------------------------------------------------------------------

    def _make_request(self, params: dict) -> Optional[requests.Response]:
        """Make HTTP GET with exponential backoff retry."""
        for attempt in range(self.max_retries):
            self._rate_limit_wait()
            try:
                response = requests.get(
                    self.base_url, params=params, timeout=self.timeout
                )
                if response.status_code == 429:
                    wait = (2 ** attempt) * 0.5
                    logger.warning(
                        "FAERS rate limited (429), waiting %.1fs (attempt %d/%d)",
                        wait, attempt + 1, self.max_retries,
                    )
                    time.sleep(wait)
                    continue
                if response.status_code == 404:
                    # No results for this query -- not an error
                    return None
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait = (2 ** attempt) * 0.5
                    logger.warning("FAERS request failed: %s, retrying in %.1fs", e, wait)
                    time.sleep(wait)
                else:
                    logger.error("FAERS request failed after %d attempts: %s", self.max_retries, e)
                    return None
        return None

    # ------------------------------------------------------------------
    # Cache key
    # ------------------------------------------------------------------

    def _cache_key(self, drug1: str, drug2: str) -> str:
        """Order-independent cache key for a drug pair."""
        pair = tuple(sorted([drug1.lower().strip(), drug2.lower().strip()]))
        raw = f"faers:{pair[0]}:{pair[1]}"
        return f"faers_{hashlib.md5(raw.encode()).hexdigest()}"

    # ------------------------------------------------------------------
    # Core query
    # ------------------------------------------------------------------

    def _build_search_query(self, drug1: str, drug2: str) -> str:
        """
        Build an openFDA search string requiring both drugs in the
        patient.drug.medicinalproduct field.
        """
        d1 = drug1.strip().lower()
        d2 = drug2.strip().lower()
        return (
            f'patient.drug.medicinalproduct:"{d1}"'
            f'+AND+patient.drug.medicinalproduct:"{d2}"'
        )

    def query_adverse_events(self, drug1: str, drug2: str) -> Optional[FaersResult]:
        """
        Main entry point -- fetch FAERS data for a drug pair.

        Returns a FaersResult dataclass or None on failure / no data.
        Results are cached for 24 hours.
        """
        cache_key = self._cache_key(drug1, drug2)
        cached = cache.get(cache_key)
        if cached is not None:
            logger.info("FAERS cache hit for '%s' + '%s'", drug1, drug2)
            return cached

        logger.info("FAERS querying adverse events for '%s' + '%s'", drug1, drug2)

        search = self._build_search_query(drug1, drug2)

        # 1. Total report count
        total_reports = self._fetch_total_count(search)
        if total_reports is None or total_reports == 0:
            # Cache the empty result so we don't re-query
            empty = FaersResult(
                drug_pair=(drug1, drug2),
                total_reports=0,
                top_reactions=[],
                seriousness_stats={},
                signal_score=0.0,
                reporter_breakdown={},
            )
            cache.set(cache_key, empty, self.cache_ttl)
            return empty

        # 2. Top adverse reactions
        top_reactions = self._fetch_top_reactions(search)

        # 3. Seriousness breakdown
        seriousness = self._fetch_seriousness(search)

        # 4. Reporter types
        reporters = self._fetch_reporter_types(search)

        # 5. Compute signal score
        signal = self._compute_signal_score(total_reports, seriousness)

        result = FaersResult(
            drug_pair=(drug1, drug2),
            total_reports=total_reports,
            top_reactions=top_reactions,
            seriousness_stats=seriousness,
            signal_score=signal,
            reporter_breakdown=reporters,
            query_url=f"{self.base_url}?search={search}&limit=1",
        )

        cache.set(cache_key, result, self.cache_ttl)
        logger.info(
            "FAERS result for '%s' + '%s': %d reports, signal=%.3f",
            drug1, drug2, total_reports, signal,
        )
        return result

    # ------------------------------------------------------------------
    # Sub-queries
    # ------------------------------------------------------------------

    def _fetch_total_count(self, search: str) -> Optional[int]:
        """Retrieve total matching report count."""
        resp = self._make_request({'search': search, 'limit': 1})
        if resp is None:
            return 0
        try:
            data = resp.json()
            total = data.get('meta', {}).get('results', {}).get('total', 0)
            return int(total)
        except Exception as e:
            logger.error("FAERS count parse error: %s", e)
            return 0

    def _fetch_top_reactions(self, search: str, limit: int = 10) -> List[Dict]:
        """Fetch top adverse reactions by count."""
        resp = self._make_request({
            'search': search,
            'count': 'patient.reaction.reactionmeddrapt.exact',
            'limit': limit,
        })
        if resp is None:
            return []
        try:
            data = resp.json()
            results = data.get('results', [])
            return [
                {'reaction': r.get('term', ''), 'count': r.get('count', 0)}
                for r in results
            ]
        except Exception as e:
            logger.error("FAERS reactions parse error: %s", e)
            return []

    def _fetch_seriousness(self, search: str) -> Dict[str, int]:
        """Fetch seriousness breakdown (death, hospitalization, etc.)."""
        stats: Dict[str, int] = {}
        fields = {
            'death': 'seriousnessdeath',
            'hospitalization': 'seriousnesshospitalization',
            'life_threatening': 'seriousnesslifethreatening',
            'disability': 'seriousnessdisabling',
            'congenital_anomaly': 'seriousnesscongenitalanomali',
            'other_serious': 'seriousnessother',
        }
        for label, fda_field in fields.items():
            resp = self._make_request({
                'search': search,
                'count': fda_field,
            })
            if resp is None:
                stats[label] = 0
                continue
            try:
                data = resp.json()
                results = data.get('results', [])
                # Count entries where field == 1 (yes)
                for r in results:
                    if str(r.get('term')) == '1':
                        stats[label] = r.get('count', 0)
                        break
                else:
                    stats[label] = 0
            except Exception as e:
                logger.warning("FAERS seriousness parse for %s: %s", label, e)
                stats[label] = 0
        return stats

    def _fetch_reporter_types(self, search: str) -> Dict[str, int]:
        """Fetch breakdown by reporter qualification."""
        resp = self._make_request({
            'search': search,
            'count': 'primarysource.qualification',
        })
        if resp is None:
            return {}
        try:
            data = resp.json()
            results = data.get('results', [])
            # FDA qualification codes: 1=physician, 2=pharmacist, 3=other health professional,
            # 4=lawyer, 5=consumer
            code_map = {
                '1': 'physician',
                '2': 'pharmacist',
                '3': 'other_health_professional',
                '4': 'lawyer',
                '5': 'consumer',
            }
            breakdown = {}
            for r in results:
                code = str(r.get('term', ''))
                label = code_map.get(code, f'unknown_{code}')
                breakdown[label] = r.get('count', 0)
            return breakdown
        except Exception as e:
            logger.error("FAERS reporter parse error: %s", e)
            return {}

    # ------------------------------------------------------------------
    # Signal score computation
    # ------------------------------------------------------------------

    def _compute_signal_score(self, total_reports: int, seriousness: Dict[str, int]) -> float:
        """
        Compute a 0.0-1.0 composite safety signal score based on:
        - Report volume (log-scaled, caps at ~10k reports)
        - Seriousness ratio (proportion of serious outcomes)

        Higher = stronger adverse signal.
        """
        import math

        if total_reports == 0:
            return 0.0

        # Volume component: log10 scaling, cap at 1.0 around 10k reports
        volume_score = min(1.0, math.log10(max(1, total_reports)) / 4.0)

        # Seriousness component: weighted sum of serious outcomes / total
        serious_count = (
            seriousness.get('death', 0) * 3.0
            + seriousness.get('life_threatening', 0) * 2.5
            + seriousness.get('hospitalization', 0) * 2.0
            + seriousness.get('disability', 0) * 1.5
            + seriousness.get('other_serious', 0) * 1.0
        )
        # Normalize: if weighted serious count >= total, max out at 1.0
        seriousness_ratio = min(1.0, serious_count / max(1, total_reports))

        # Weighted combination: 40% volume, 60% seriousness
        score = 0.4 * volume_score + 0.6 * seriousness_ratio
        return round(min(1.0, max(0.0, score)), 4)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_faers_instance: Optional[FaersService] = None


def get_faers_service() -> FaersService:
    """Get the singleton FaersService instance."""
    global _faers_instance
    if _faers_instance is None:
        _faers_instance = FaersService()
    return _faers_instance
