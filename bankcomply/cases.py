"""Case workflow with role checks, four-eyes approval and full auditing."""

from __future__ import annotations

from dataclasses import dataclass, field

from .audit import AuditLog
from .rbac import PermissionDenied, Principal, has_permission, mask_name, require


class WorkflowError(Exception):
    """An action is not allowed in the case's current state or breaks a process rule."""


# action -> (permission, {from_status: to_status})
ACTIONS: dict[str, tuple[str, dict[str, str]]] = {
    "start_review": ("start_review", {"OPEN": "IN_REVIEW"}),
    "escalate": ("escalate_case", {"IN_REVIEW": "ESCALATED"}),
    "close_no_action": ("close_no_action", {"IN_REVIEW": "CLOSED"}),
    "approve_sar": ("approve_sar", {"ESCALATED": "SAR_APPROVED"}),
    "close_escalated": ("close_escalated", {"ESCALATED": "CLOSED"}),
}


@dataclass
class Case:
    id: str
    customer_id: str
    customer_name: str
    opened_by: str
    status: str
    recommendation: str
    risk_score: int
    tier: str
    reasons: list[dict]
    narrative: str
    citations: list[dict]
    history: list[dict] = field(default_factory=list)
    escalated_by: str | None = None


class CaseService:
    def __init__(self, audit: AuditLog):
        self.audit = audit
        self._cases: dict[str, Case] = {}

    def _log(self, p: Principal, action: str, resource: str, outcome: str, **details):
        self.audit.append(p.user, p.role, action, resource, outcome, details)

    def _need(self, p: Principal, permission: str, action: str, resource: str):
        try:
            require(p, permission)
        except PermissionDenied:
            self._log(p, action, resource, "denied", permission=permission)
            raise

    def open_case(self, p: Principal, customer, assessment: dict, narrative: str, citations: list[dict]) -> Case:
        self._need(p, "open_case", "case.open", customer.id)
        case = Case(
            id=f"CASE-{len(self._cases) + 1:05d}", customer_id=customer.id, customer_name=customer.name,
            opened_by=p.user, status="OPEN", recommendation=assessment["recommendation"],
            risk_score=assessment["score"], tier=assessment["tier"], reasons=assessment["reasons"],
            narrative=narrative, citations=citations,
        )
        self._cases[case.id] = case
        self._log(p, "case.open", case.id, "ok", customer=customer.id, recommendation=case.recommendation,
                  score=case.risk_score)
        return case

    def _get(self, case_id: str) -> Case:
        if case_id not in self._cases:
            raise KeyError(case_id)
        return self._cases[case_id]

    def _render(self, p: Principal, case: Case) -> dict:
        d = {k: getattr(case, k) for k in (
            "id", "customer_id", "opened_by", "status", "recommendation", "risk_score", "tier", "reasons",
            "narrative", "citations", "history", "escalated_by")}
        d["customer_name"] = case.customer_name if has_permission(p, "view_pii") else mask_name(case.customer_name)
        if not has_permission(p, "view_pii"):
            d["narrative"] = d["narrative"].replace(case.customer_name, d["customer_name"])
        return d

    def view_case(self, p: Principal, case_id: str) -> dict:
        self._need(p, "view_case", "case.view", case_id)
        case = self._get(case_id)
        self._log(p, "case.view", case_id, "ok", pii_visible=has_permission(p, "view_pii"))
        return self._render(p, case)

    def list_cases(self, p: Principal, status: str | None = None) -> list[dict]:
        self._need(p, "view_case", "case.list", "*")
        self._log(p, "case.list", "*", "ok", status=status)
        return [self._render(p, c) for c in self._cases.values() if status in (None, c.status)]

    def act(self, p: Principal, case_id: str, action: str, note: str = "") -> dict:
        if action not in ACTIONS:
            raise WorkflowError(f"unknown action '{action}'")
        case = self._get(case_id)
        permission, transitions = ACTIONS[action]
        # Closing a case the system recommended for SAR needs the officer-level permission.
        if action == "close_no_action" and case.recommendation == "ESCALATE_SAR_RECOMMENDED":
            permission = "close_escalated"
        self._need(p, permission, f"case.{action}", case_id)

        def reject(reason: str):
            self._log(p, f"case.{action}", case_id, "rejected", reason=reason)
            raise WorkflowError(reason)

        if case.status not in transitions:
            reject(f"cannot {action} a case that is {case.status}")
        if action != "start_review" and not note.strip():
            reject(f"a note is required to {action}")
        if action == "approve_sar" and p.user in (case.opened_by, case.escalated_by):
            reject("four-eyes rule: the approver must differ from the preparer and the escalating analyst")

        previous, case.status = case.status, transitions[case.status]
        if action == "escalate":
            case.escalated_by = p.user
        case.history.append({"ts": self.audit.now(), "actor": p.user, "action": action, "from": previous,
                             "to": case.status, "note": note})
        self._log(p, f"case.{action}", case_id, "ok", **{"from": previous, "to": case.status})
        return self._render(p, case)
