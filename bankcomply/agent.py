"""The KYC/AML case agent.

A fixed, auditable tool pipeline: profile -> KYC checks -> watchlist screening -> transaction
history -> AML rules -> risk assessment -> policy lookup -> narrative -> (optionally) open case.
Each tool call is permission-checked and written to the audit log. The agent only *recommends*:
approving a SAR is a separate, human, four-eyes step in ``CaseService``.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Callable

from policyguard import HybridRetriever, split_clauses

from .aml_rules import AmlConfig, run_rules
from .audit import AuditLog
from .cases import CaseService
from .data import Dataset
from .kyc import kyc_findings, screen_name
from .rbac import PermissionDenied, Principal, require
from .scoring import assess

DEFAULT_POLICY = Path(__file__).resolve().parent.parent / "data" / "policies" / "bank_aml_policy.md"

# reason key -> query used to find the supporting policy clause
_POLICY_QUERIES = {
    "structuring": "structuring cash deposits below reporting threshold",
    "rapid_movement": "rapid movement of funds pass-through layering",
    "high_risk_geo": "high-risk jurisdictions counterparties",
    "watchlist": "watchlist sanctions name match",
    "pep": "politically exposed persons enhanced due diligence",
    "velocity": "unusual transaction volume above expected profile",
    "large_cash": "large cash transactions reporting threshold",
    "doc_expired": "expired identity document KYC refresh",
    "review_overdue": "overdue KYC review refresh records",
}
_NUMBER = re.compile(r"\d[\d,]*\.?\d*")


def _numbers(text: str) -> set[str]:
    out = set()
    for m in _NUMBER.findall(text):
        m = m.replace(",", "").rstrip(".")
        if len(m.replace(".", "")) >= 2:      # ignore single digits
            out.add(m)
    return out


def verify_narrative(narrative: str, facts: dict) -> list[str]:
    """Return numbers in ``narrative`` that appear nowhere in the structured ``facts``."""
    allowed = _numbers(json.dumps(facts, default=str))
    return sorted(n for n in _numbers(narrative) if n not in allowed)


class CaseAgent:
    def __init__(self, data: Dataset, service: CaseService, policy_path: Path | str = DEFAULT_POLICY,
                 config: AmlConfig | None = None, narrative_writer: Callable[[dict], str] | None = None):
        self.data, self.service, self.audit = data, service, service.audit
        self.config = config or AmlConfig()
        self.narrative_writer = narrative_writer
        self.policy = HybridRetriever(split_clauses(Path(policy_path).read_text(encoding="utf-8")))

    # -- tool plumbing ------------------------------------------------------------------
    def _tool(self, p: Principal, trace: list, name: str, permission: str, resource: str, fn):
        try:
            require(p, permission)
        except PermissionDenied:
            self.audit.append(p.user, p.role, f"agent.{name}", resource, "denied", {"permission": permission})
            raise
        result, summary = fn()
        trace.append({"tool": name, "summary": summary})
        self.audit.append(p.user, p.role, f"agent.{name}", resource, "ok", {"summary": summary})
        return result

    def _cite(self, key: str) -> dict | None:
        query = _POLICY_QUERIES.get(key)
        hits = self.policy.search(query, k=1) if query else []
        if not hits:
            return None
        c = hits[0].clause
        return {"reason": key, "clause_id": c.id, "heading": c.heading}

    def _narrative(self, p: Principal, customer, assessment: dict, citations: list[dict]) -> str:
        facts = {"customer_id": customer.id, "assessment": assessment, "policy": citations}
        template = [f"Customer {customer.name} ({customer.id}): risk score {assessment['score']}/100 ({assessment['tier']}), "
                    f"recommendation {assessment['recommendation']}."]
        template += [f"- {r['text']} (+{r['points']})" for r in assessment["reasons"]]
        template += [f"Policy basis: section {c['clause_id']} {c['heading']}." for c in citations]
        default = "\n".join(template)
        if self.narrative_writer is None:
            return default
        text = self.narrative_writer(facts)
        unsupported = verify_narrative(text, facts)
        if unsupported:   # a model-written narrative may not introduce figures that are not in the evidence
            self.audit.append(p.user, p.role, "agent.narrative", customer.id, "rejected",
                              {"unsupported_numbers": unsupported})
            return default
        return text

    # -- main entry point ---------------------------------------------------------------
    def investigate(self, p: Principal, customer_id: str, today: date | None = None, open_case: bool = True) -> dict:
        today = today or self.data.end
        trace: list[dict] = []
        customer = self._tool(p, trace, "get_customer_profile", "view_customer", customer_id, lambda: (
            self.data.customers[customer_id], f"loaded profile for {customer_id}"))
        kyc = self._tool(p, trace, "kyc_checks", "view_customer", customer_id, lambda: (
            (f := kyc_findings(customer, today)), f"{len(f)} KYC finding(s)"))
        matches = self._tool(p, trace, "screen_watchlist", "run_screening", customer_id, lambda: (
            (m := screen_name(customer.name, self.data.watchlist)), f"{len(m)} watchlist match(es)"))
        txns = self._tool(p, trace, "get_transactions", "view_transactions", customer_id, lambda: (
            (t := self.data.transactions_for(customer_id)), f"{len(t)} transaction(s)"))
        alerts = self._tool(p, trace, "run_aml_rules", "run_screening", customer_id, lambda: (
            (a := run_rules(customer, txns, self.config)), f"{len(a)} alert(s): {', '.join(x.rule for x in a) or 'none'}"))
        risk = assess(matches, kyc, alerts).to_dict()

        def lookup():
            seen, found = set(), []
            for r in risk["reasons"]:
                c = self._cite(r["key"])
                if c and c["clause_id"] not in seen:
                    seen.add(c["clause_id"])
                    found.append(c)
            return found, f"{len(found)} policy clause(s) cited"
        citations = self._tool(p, trace, "lookup_policy", "view_policy", customer_id, lookup)

        narrative = self._narrative(p, customer, risk, citations)
        case_id = None
        if open_case and risk["recommendation"] != "NO_ACTION":
            case = self.service.open_case(p, customer, risk, narrative, citations)
            case_id = case.id
        else:
            self.audit.append(p.user, p.role, "agent.decision", customer_id, "ok",
                              {"recommendation": risk["recommendation"], "case_opened": False})
        return {"customer_id": customer_id, "recommendation": risk["recommendation"], "risk": risk,
                "citations": citations, "narrative": narrative, "case_id": case_id, "trace": trace}
