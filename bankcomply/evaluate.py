"""Compare agent recommendations with the ground-truth labels of a labelled dataset."""

from __future__ import annotations

from collections import Counter

_ACTION = {"ESCALATE_SAR_RECOMMENDED": "SAR", "ENHANCED_DUE_DILIGENCE": "EDD", "NO_ACTION": "NONE"}


def evaluate(truth: dict[str, dict], results: dict[str, dict]) -> dict:
    """``results`` maps customer_id -> investigate() output."""
    confusion: Counter = Counter()
    for cid, result in results.items():          # customers without a label are expected to be clean
        expected = truth.get(cid, {"expected": "NONE"})["expected"]
        confusion[(expected, _ACTION[result["recommendation"]])] += 1
    total = sum(confusion.values())
    correct = sum(n for (e, p), n in confusion.items() if e == p)
    tp = confusion[("SAR", "SAR")]
    predicted_sar = sum(n for (e, p), n in confusion.items() if p == "SAR")
    expected_sar = sum(n for (e, p), n in confusion.items() if e == "SAR")
    alerted = lambda e: sum(n for (ee, p), n in confusion.items() if ee == e and p != "NONE")   # noqa: E731
    expected_edd_or_sar = expected_sar + sum(n for (e, p), n in confusion.items() if e == "EDD")
    return {
        "customers": total,
        "accuracy": correct / total if total else 0.0,
        "sar_precision": tp / predicted_sar if predicted_sar else 0.0,
        "sar_recall": tp / expected_sar if expected_sar else 0.0,
        "any_alert_recall": (alerted("SAR") + alerted("EDD")) / expected_edd_or_sar if expected_edd_or_sar else 0.0,
        "false_alerts_on_clean": sum(n for (e, p), n in confusion.items() if e == "NONE" and p != "NONE"),
        "confusion": {f"expected_{e}->got_{p}": n for (e, p), n in sorted(confusion.items())},
    }
