"""Rule-based AML transaction monitoring. Every alert carries the transactions that caused it."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from .data import HIGH_RISK_COUNTRIES, Customer, Transaction


@dataclass(frozen=True)
class AmlConfig:
    reporting_threshold: float = 10_000.0
    structuring_min_ratio: float = 0.80
    structuring_window_days: int = 7
    structuring_min_count: int = 3
    rapid_min_inflow: float = 20_000.0
    rapid_outflow_ratio: float = 0.90
    rapid_window_hours: int = 48
    velocity_window_days: int = 7
    velocity_floor: float = 25_000.0
    velocity_multiple: float = 2.0
    high_risk_countries: frozenset = HIGH_RISK_COUNTRIES
    geo_min_count: int = 2
    geo_medium_total: float = 10_000.0
    geo_high_total: float = 25_000.0


@dataclass(frozen=True)
class Alert:
    rule: str
    severity: str                 # low | medium | high
    description: str
    tx_ids: tuple[str, ...] = ()
    evidence: dict = field(default_factory=dict)


def _structuring(txns: list[Transaction], cfg: AmlConfig) -> list[Alert]:
    lo = cfg.structuring_min_ratio * cfg.reporting_threshold
    cash = [t for t in txns if t.channel == "cash" and t.direction == "in" and lo <= t.amount < cfg.reporting_threshold]
    window = timedelta(days=cfg.structuring_window_days)
    best: list[Transaction] = []
    j = 0
    for i in range(len(cash)):
        while cash[i].ts - cash[j].ts > window:
            j += 1
        if i - j + 1 > len(best):
            best = cash[j:i + 1]
    if len(best) >= cfg.structuring_min_count:
        total = sum(t.amount for t in best)
        return [Alert("structuring", "high",
                      f"{len(best)} cash deposits of {lo:,.0f} to {cfg.reporting_threshold:,.0f} within "
                      f"{cfg.structuring_window_days} days, totalling {total:,.2f}, each just under the reporting threshold.",
                      tuple(t.id for t in best), {"count": len(best), "total": round(total, 2)})]
    return []


def _rapid_movement(txns: list[Transaction], cfg: AmlConfig) -> list[Alert]:
    moving = ("wire", "transfer")
    window = timedelta(hours=cfg.rapid_window_hours)
    for t in txns:
        if t.direction != "in" or t.channel not in moving or t.amount < cfg.rapid_min_inflow:
            continue
        outs = [o for o in txns if o.direction == "out" and o.channel in moving and t.ts < o.ts <= t.ts + window]
        out_total = sum(o.amount for o in outs)
        if out_total >= cfg.rapid_outflow_ratio * t.amount:
            return [Alert("rapid_movement", "high",
                          f"Inflow of {t.amount:,.2f} was followed by outflows of {out_total:,.2f} within "
                          f"{cfg.rapid_window_hours} hours ({out_total / t.amount:.0%} of the inflow).",
                          (t.id, *[o.id for o in outs]), {"inflow": t.amount, "outflow": round(out_total, 2)})]
    return []


def _velocity(customer: Customer, txns: list[Transaction], cfg: AmlConfig) -> list[Alert]:
    window = timedelta(days=cfg.velocity_window_days)
    limit = max(cfg.velocity_floor, cfg.velocity_multiple * customer.expected_monthly_volume)
    best_sum, best_range, running, j = 0.0, (0, 0), 0.0, 0
    for i, t in enumerate(txns):
        running += t.amount
        while t.ts - txns[j].ts > window:
            running -= txns[j].amount
            j += 1
        if running > best_sum:
            best_sum, best_range = running, (j, i)
    if best_sum > limit:
        ids = tuple(t.id for t in txns[best_range[0]:best_range[1] + 1])
        return [Alert("velocity", "medium",
                      f"{best_sum:,.2f} moved within {cfg.velocity_window_days} days, above the limit of {limit:,.2f} "
                      f"for this customer's profile.", ids, {"window_total": round(best_sum, 2), "limit": round(limit, 2)})]
    return []


def _high_risk_geo(txns: list[Transaction], cfg: AmlConfig) -> list[Alert]:
    hits = [t for t in txns if t.counterparty_country in cfg.high_risk_countries]
    total = sum(t.amount for t in hits)
    if len(hits) >= cfg.geo_min_count and total >= cfg.geo_medium_total:
        severity = "high" if total >= cfg.geo_high_total else "medium"
        countries = ", ".join(sorted({t.counterparty_country for t in hits}))
        return [Alert("high_risk_geo", severity,
                      f"{len(hits)} transactions totalling {total:,.2f} with high-risk jurisdictions ({countries}).",
                      tuple(t.id for t in hits), {"count": len(hits), "total": round(total, 2)})]
    return []


def _large_cash(txns: list[Transaction], cfg: AmlConfig) -> list[Alert]:
    big = [t for t in txns if t.channel == "cash" and t.amount >= cfg.reporting_threshold]
    if big:
        return [Alert("large_cash", "low",
                      f"{len(big)} cash transaction(s) at or above the {cfg.reporting_threshold:,.0f} reporting threshold "
                      f"(largest {max(t.amount for t in big):,.2f}).", tuple(t.id for t in big),
                      {"count": len(big)})]
    return []


def run_rules(customer: Customer, txns: list[Transaction], cfg: AmlConfig | None = None) -> list[Alert]:
    cfg = cfg or AmlConfig()
    txns = sorted(txns, key=lambda t: t.ts)
    if not txns:
        return []
    return (_structuring(txns, cfg) + _rapid_movement(txns, cfg) + _velocity(customer, txns, cfg)
            + _high_risk_geo(txns, cfg) + _large_cash(txns, cfg))
