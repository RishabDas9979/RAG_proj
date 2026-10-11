"""End-to-end demo: investigate every customer, evaluate, then walk one case through approval.

Run:  python -m bankcomply.demo
"""

from __future__ import annotations

import json

from .agent import CaseAgent
from .audit import AuditLog
from .cases import CaseService, WorkflowError
from .data import generate_dataset
from .evaluate import evaluate
from .rbac import PermissionDenied, Principal


def main() -> None:
    # Same labels, two behaviours: "plain" typologies sit right on the rule thresholds,
    # "evasive" ones try to slip past them. Only the plain run is a sanity check; the evasive
    # run shows where the rules are weak.
    evasive_data = generate_dataset(evasive=True)
    evasive_agent = CaseAgent(evasive_data, CaseService(AuditLog()))
    evasive_results = {cid: evasive_agent.investigate(Principal("asha", "analyst"), cid) for cid in evasive_data.customers}
    print("EVASIVE data (suspicious customers avoid exact thresholds):")
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in evaluate(evasive_data.truth, evasive_results).items()}, indent=2))

    data = generate_dataset()
    audit = AuditLog()
    service = CaseService(audit)
    agent = CaseAgent(data, service)
    analyst = Principal("asha", "analyst")
    officer = Principal("ravi", "compliance_officer")

    results = {cid: agent.investigate(analyst, cid) for cid in data.customers}
    metrics = evaluate(data.truth, results)
    print(f"\nPLAIN data: investigated {metrics['customers']} customers, opened {len(service.list_cases(officer))} cases")
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()}, indent=2))

    sar = next(r for r in results.values() if r["recommendation"] == "ESCALATE_SAR_RECOMMENDED")
    print("\n--- Example case ---")
    print(sar["narrative"])
    cid = sar["case_id"]

    service.act(analyst, cid, "start_review")
    service.act(analyst, cid, "escalate", "Pattern matches procedure; escalating to the officer.")
    for who, label in ((analyst, "analyst tries to approve the SAR"), (Principal("asha", "compliance_officer"), "preparer promoted to officer")):
        try:
            service.act(who, cid, "approve_sar", "approve")
        except (PermissionDenied, WorkflowError) as exc:
            print(f"\nBlocked ({label}): {exc}")
    done = service.act(officer, cid, "approve_sar", "Reviewed evidence; SAR to be filed.")
    print(f"\nOfficer approved: status={done['status']}")
    print(f"Auditor view masks the name: {service.view_case(Principal('meera', 'auditor'), cid)['customer_name']}")

    ok, bad = audit.verify()
    print(f"\nAudit log: {len(audit)} entries, chain intact={ok}")


if __name__ == "__main__":
    main()
