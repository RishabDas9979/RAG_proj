"""Explainable risk scoring. Every point is attributed to a named reason with evidence."""

from __future__ import annotations

from dataclasses import dataclass, field

from .aml_rules import Alert
from .kyc import KycFinding, WatchlistMatch

ALERT_POINTS = {"high": 35, "medium": 25, "low": 3}
KYC_POINTS = {"PEP": 25, "HIGH_RISK_COUNTRY": 15, "DOC_EXPIRED": 5, "REVIEW_OVERDUE": 5, "DOC_EXPIRING": 1}
WATCHLIST_POINTS = {"strong": 70, "possible": 30}
HIGH_TIER, MEDIUM_TIER = 60, 25


@dataclass(frozen=True)
class Reason:
    key: str            # used to look up the supporting policy clause
    points: int
    text: str
    tx_ids: tuple[str, ...] = ()


@dataclass
class RiskAssessment:
    score: int
    tier: str
    recommendation: str
    reasons: list[Reason] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": self.score, "tier": self.tier, "recommendation": self.recommendation,
            "reasons": [{"key": r.key, "points": r.points, "text": r.text, "tx_ids": list(r.tx_ids)} for r in self.reasons],
        }


def assess(matches: list[WatchlistMatch], kyc: list[KycFinding], alerts: list[Alert]) -> RiskAssessment:
    reasons: list[Reason] = []
    if matches:
        top = matches[0]
        reasons.append(Reason("watchlist", WATCHLIST_POINTS[top.strength],
                              f"{top.strength.capitalize()} watchlist name match to '{top.matched_name}' on {top.list_name} "
                              f"(similarity {top.score:.2f})."))
    for f in kyc:
        reasons.append(Reason(f.code.lower(), KYC_POINTS.get(f.code, 0), f.text))
    seen: set[str] = set()
    for a in sorted(alerts, key=lambda a: -ALERT_POINTS[a.severity]):
        if a.rule in seen:     # one scoring contribution per rule, however many times it fired
            continue
        seen.add(a.rule)
        reasons.append(Reason(a.rule, ALERT_POINTS[a.severity], a.description, a.tx_ids))

    score = min(100, sum(r.points for r in reasons))
    tier = "high" if score >= HIGH_TIER else "medium" if score >= MEDIUM_TIER else "low"
    strong = bool(matches) and matches[0].strength == "strong"
    if tier == "high" and (strong or any(a.severity == "high" for a in alerts)):
        recommendation = "ESCALATE_SAR_RECOMMENDED"
    elif tier in ("high", "medium"):
        recommendation = "ENHANCED_DUE_DILIGENCE"
    else:
        recommendation = "NO_ACTION"
    return RiskAssessment(score, tier, recommendation, [r for r in reasons if r.points > 0])
