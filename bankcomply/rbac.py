"""Role-based access control with separation of duties."""

from __future__ import annotations

from dataclasses import dataclass


class PermissionDenied(Exception):
    def __init__(self, principal: "Principal", permission: str):
        super().__init__(f"{principal.user} ({principal.role}) lacks permission '{permission}'")
        self.principal = principal
        self.permission = permission


_ANALYST = frozenset({
    "view_customer", "view_transactions", "run_screening", "view_policy", "open_case",
    "start_review", "escalate_case", "view_case", "view_pii",
})
_SENIOR = _ANALYST | {"close_no_action"}
_OFFICER = _SENIOR | {"approve_sar", "close_escalated", "view_audit"}

PERMISSIONS: dict[str, frozenset[str]] = {
    "analyst": _ANALYST,
    "senior_analyst": frozenset(_SENIOR),
    "compliance_officer": frozenset(_OFFICER),
    # Auditors read cases (PII masked) and the audit log, and cannot change anything.
    "auditor": frozenset({"view_case", "view_audit", "view_policy"}),
    # Admins manage users and read the audit log, but see no case or customer data.
    "admin": frozenset({"view_audit", "manage_users"}),
}


@dataclass(frozen=True)
class Principal:
    user: str
    role: str

    def __post_init__(self):
        if self.role not in PERMISSIONS:
            raise ValueError(f"unknown role: {self.role}")


def has_permission(principal: Principal, permission: str) -> bool:
    return permission in PERMISSIONS[principal.role]


def require(principal: Principal, permission: str) -> None:
    if not has_permission(principal, permission):
        raise PermissionDenied(principal, permission)


def mask_name(name: str) -> str:
    """'Zelmar Dravenko' -> 'Z***** D*******' (first letter kept)."""
    return " ".join(w[0] + "*" * (len(w) - 1) if w else w for w in name.split(" "))
