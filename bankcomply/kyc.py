"""KYC checks and fuzzy watchlist screening."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher

from .data import HIGH_RISK_COUNTRIES, Customer, WatchlistEntry

STRONG_MATCH = 0.92
POSSIBLE_MATCH = 0.85


def _tokens(name: str) -> list[str]:
    return sorted(re.findall(r"[a-z]+", name.lower()))


def name_similarity(a: str, b: str) -> float:
    """Token-sorted string similarity (0-1). Robust to word order and punctuation."""
    ta, tb = " ".join(_tokens(a)), " ".join(_tokens(b))
    if not ta or not tb:
        return 0.0
    return SequenceMatcher(None, ta, tb).ratio()


@dataclass(frozen=True)
class WatchlistMatch:
    entry_id: str
    list_name: str
    matched_name: str
    score: float

    @property
    def strength(self) -> str:
        return "strong" if self.score >= STRONG_MATCH else "possible"


def screen_name(name: str, watchlist: list[WatchlistEntry], threshold: float = POSSIBLE_MATCH) -> list[WatchlistMatch]:
    matches = []
    for entry in watchlist:
        best_name, best = max(((n, name_similarity(name, n)) for n in (entry.name, *entry.aliases)), key=lambda x: x[1])
        if best >= threshold:
            matches.append(WatchlistMatch(entry.id, entry.list_name, best_name, round(best, 3)))
    return sorted(matches, key=lambda m: -m.score)


@dataclass(frozen=True)
class KycFinding:
    code: str
    severity: str
    text: str


def kyc_findings(customer: Customer, today: date) -> list[KycFinding]:
    out: list[KycFinding] = []
    days_left = (customer.doc_expiry - today).days
    if days_left < 0:
        out.append(KycFinding("DOC_EXPIRED", "medium", f"Identity document expired {-days_left} days ago."))
    elif days_left <= 30:
        out.append(KycFinding("DOC_EXPIRING", "low", f"Identity document expires in {days_left} days."))
    since = (today - customer.last_review).days
    if since > 365:
        out.append(KycFinding("REVIEW_OVERDUE", "medium", f"Last KYC review was {since} days ago."))
    if customer.pep:
        out.append(KycFinding("PEP", "high", "Customer is a politically exposed person: enhanced due diligence applies."))
    if customer.country in HIGH_RISK_COUNTRIES:
        out.append(KycFinding("HIGH_RISK_COUNTRY", "high", f"Customer is based in a high-risk jurisdiction ({customer.country})."))
    return out
