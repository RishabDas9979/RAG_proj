import csv
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

from bankcomply import (AuditLog, AmlConfig, CaseAgent, CaseService, Customer, PermissionDenied, Principal,
                        Transaction, generate_dataset, load_paysim_csv, run_rules)
from bankcomply.agent import verify_narrative
from bankcomply.api import make_server
from bankcomply.cases import WorkflowError
from bankcomply.data import WATCHLIST
from bankcomply.evaluate import evaluate
from bankcomply.kyc import kyc_findings, name_similarity, screen_name
from bankcomply.rbac import has_permission, mask_name
from bankcomply.scoring import assess

ANALYST = Principal("asha", "analyst")
SENIOR = Principal("sena", "senior_analyst")
OFFICER = Principal("ravi", "compliance_officer")
AUDITOR = Principal("meera", "auditor")
ADMIN = Principal("dev", "admin")


def customer(**kw):
    base = dict(id="C1", name="Test Person", country="IN", occupation="x", pep=False,
                doc_expiry=date(2030, 1, 1), last_review=date(2026, 6, 1), expected_monthly_volume=5000.0)
    return Customer(**{**base, **kw})


def tx(i, day, channel, direction, amount, country="IN", hours=0):
    return Transaction(f"T{i}", "C1", datetime(2026, 9, 1) + timedelta(days=day, hours=hours), channel, direction,
                       amount, country, "cp")


class RbacTests(unittest.TestCase):
    def test_permission_matrix(self):
        self.assertTrue(has_permission(ANALYST, "escalate_case"))
        self.assertFalse(has_permission(ANALYST, "approve_sar"))
        self.assertFalse(has_permission(SENIOR, "approve_sar"))
        self.assertTrue(has_permission(OFFICER, "approve_sar"))
        self.assertFalse(has_permission(AUDITOR, "open_case"))
        self.assertFalse(has_permission(ADMIN, "view_case"))      # separation of duties
        self.assertTrue(has_permission(ADMIN, "view_audit"))

    def test_unknown_role_rejected(self):
        with self.assertRaises(ValueError):
            Principal("x", "root")

    def test_mask_name(self):
        self.assertEqual(mask_name("Zelmar Dravenko"), "Z***** D*******")


class AuditTests(unittest.TestCase):
    def make(self):
        log = AuditLog(clock=lambda: "2026-10-01T00:00:00+00:00")
        for i in range(5):
            log.append("u", "analyst", "act", f"r{i}", "ok", {"i": i})
        return log

    def test_chain_verifies(self):
        self.assertEqual(self.make().verify(), (True, None))

    def test_update_and_delete_blocked_by_triggers(self):
        import sqlite3
        log = self.make()
        with self.assertRaises(sqlite3.DatabaseError):
            log._conn.execute("UPDATE entries SET outcome='ok' WHERE seq=2")
        with self.assertRaises(sqlite3.DatabaseError):
            log._conn.execute("DELETE FROM entries WHERE seq=2")

    def test_tampering_detected_even_if_triggers_removed(self):
        log = self.make()
        log._conn.execute("DROP TRIGGER entries_no_update")
        log._conn.execute("UPDATE entries SET outcome='ok', actor='mallory' WHERE seq=3")
        self.assertEqual(log.verify(), (False, 3))

    def test_deleted_middle_entry_detected(self):
        log = self.make()
        log._conn.execute("DROP TRIGGER entries_no_delete")
        log._conn.execute("DELETE FROM entries WHERE seq=3")
        ok, bad = log.verify()
        self.assertFalse(ok)

    def test_truncation_needs_recorded_head(self):
        log = self.make()
        head = log.head()
        log._conn.execute("DROP TRIGGER entries_no_delete")
        log._conn.execute("DELETE FROM entries WHERE seq=5")
        self.assertTrue(log.verify()[0])                         # chain alone cannot see tail removal
        self.assertFalse(log.verify(expected_head=head)[0])      # an externally stored head can

    def test_empty_log_is_valid(self):
        self.assertEqual(AuditLog().verify(), (True, None))


class RuleTests(unittest.TestCase):
    cfg = AmlConfig()

    def rules(self, txns, **kw):
        return {a.rule: a for a in run_rules(customer(**kw), txns, self.cfg)}

    def test_structuring_detected(self):
        txns = [tx(i, i, "cash", "in", 9000 + i * 100) for i in range(4)]
        self.assertIn("structuring", self.rules(txns))

    def test_two_deposits_not_structuring(self):
        self.assertNotIn("structuring", self.rules([tx(1, 0, "cash", "in", 9500), tx(2, 1, "cash", "in", 9400)]))

    def test_deposits_far_apart_not_structuring(self):
        txns = [tx(i, i * 10, "cash", "in", 9500) for i in range(4)]
        self.assertNotIn("structuring", self.rules(txns))

    def test_rapid_movement(self):
        txns = [tx(1, 0, "wire", "in", 50000), tx(2, 0, "wire", "out", 48000, hours=10)]
        self.assertEqual(self.rules(txns)["rapid_movement"].tx_ids, ("T1", "T2"))

    def test_inflow_kept_is_not_rapid_movement(self):
        txns = [tx(1, 0, "wire", "in", 50000), tx(2, 0, "wire", "out", 5000, hours=10)]
        self.assertNotIn("rapid_movement", self.rules(txns))

    def test_outflow_after_window_is_not_rapid_movement(self):
        txns = [tx(1, 0, "wire", "in", 50000), tx(2, 4, "wire", "out", 49000)]
        self.assertNotIn("rapid_movement", self.rules(txns))

    def test_velocity_uses_customer_profile(self):
        txns = [tx(i, 0, "transfer", "in", 4000, hours=i) for i in range(10)]   # 40k in a day
        self.assertIn("velocity", self.rules(txns, expected_monthly_volume=5000.0))
        self.assertNotIn("velocity", self.rules(txns, expected_monthly_volume=40000.0))

    def test_high_risk_geo_severity_scales(self):
        small = [tx(i, i, "wire", "out", 6000, "XA") for i in range(2)]
        big = [tx(i, i, "wire", "out", 15000, "XB") for i in range(2)]
        self.assertEqual(self.rules(small)["high_risk_geo"].severity, "medium")
        self.assertEqual(self.rules(big)["high_risk_geo"].severity, "high")
        self.assertNotIn("high_risk_geo", self.rules([tx(1, 0, "wire", "out", 90000, "XA")]))  # single txn

    def test_large_cash_is_low_severity(self):
        self.assertEqual(self.rules([tx(1, 0, "cash", "in", 12000)])["large_cash"].severity, "low")

    def test_no_transactions(self):
        self.assertEqual(run_rules(customer(), []), [])


class KycTests(unittest.TestCase):
    def test_watchlist_variants(self):
        for name in ("Zelmar Dravenko", "Dravenko Zelmar", "Zelmar Dravenco"):
            m = screen_name(name, WATCHLIST)
            self.assertTrue(m and m[0].strength == "strong", name)

    def test_near_misses_not_flagged(self):
        for name in ("Kasim Oduya", "Zelmar Dravis", "Priya Sharma"):
            self.assertEqual(screen_name(name, WATCHLIST), [], name)

    def test_similarity_bounds(self):
        self.assertEqual(name_similarity("", "abc"), 0.0)
        self.assertEqual(name_similarity("A B", "b a"), 1.0)

    def test_kyc_findings(self):
        codes = {f.code for f in kyc_findings(
            customer(pep=True, doc_expiry=date(2026, 9, 1), last_review=date(2025, 1, 1), country="XA"), date(2026, 10, 1))}
        self.assertEqual(codes, {"PEP", "DOC_EXPIRED", "REVIEW_OVERDUE", "HIGH_RISK_COUNTRY"})


class ScoringTests(unittest.TestCase):
    def test_clean_customer_no_action(self):
        self.assertEqual(assess([], [], []).recommendation, "NO_ACTION")

    def test_strong_match_alone_escalates(self):
        m = screen_name("Zelmar Dravenko", WATCHLIST)
        self.assertEqual(assess(m, [], []).recommendation, "ESCALATE_SAR_RECOMMENDED")

    def test_pep_alone_requires_edd_not_sar(self):
        r = assess([], kyc_findings(customer(pep=True), date(2026, 10, 1)), [])
        self.assertEqual(r.recommendation, "ENHANCED_DUE_DILIGENCE")

    def test_repeated_alerts_of_one_rule_count_once(self):
        a = run_rules(customer(), [tx(i, 0, "cash", "in", 11000, hours=i) for i in range(5)])
        self.assertEqual(sum(1 for r in assess([], [], a).reasons if r.key == "large_cash"), 1)

    def test_score_is_sum_of_reason_points(self):
        r = assess([], kyc_findings(customer(pep=True), date(2026, 10, 1)), [])
        self.assertEqual(r.score, sum(x.points for x in r.reasons))


class DataTests(unittest.TestCase):
    def test_deterministic(self):
        a, b = generate_dataset(seed=3), generate_dataset(seed=3)
        self.assertEqual([t.amount for t in a.transactions], [t.amount for t in b.transactions])

    def test_too_few_customers(self):
        with self.assertRaises(ValueError):
            generate_dataset(n_customers=10)

    def test_paysim_loader(self):
        rows = [
            ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrg", "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud"],
            [1, "TRANSFER", 181.0, "C100", 181, 0, "C200", 0, 0, 1],
            [2, "CASH_OUT", 500.0, "C300", 900, 400, "C400", 0, 0, 0],
            [3, "PAYMENT", 30.5, "C300", 400, 369.5, "M500", 0, 0, 0],
        ]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "paysim.csv"
            with open(path, "w", newline="") as fh:
                csv.writer(fh).writerows(rows)
            ds = load_paysim_csv(path)
        self.assertEqual(set(ds.customers), {"C100", "C300"})
        self.assertEqual(ds.truth["C100"]["expected"], "SAR")
        self.assertEqual(ds.truth["C300"]["expected"], "NONE")
        self.assertEqual(len(ds.transactions_for("C300")), 2)


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = generate_dataset()
        cls.audit = AuditLog()
        cls.service = CaseService(cls.audit)
        cls.agent = CaseAgent(cls.data, cls.service)
        cls.results = {cid: cls.agent.investigate(ANALYST, cid) for cid in cls.data.customers}

    def test_plain_dataset_sanity(self):
        m = evaluate(self.data.truth, self.results)
        self.assertEqual(m["sar_recall"], 1.0)
        self.assertEqual(m["sar_precision"], 1.0)
        self.assertEqual(m["false_alerts_on_clean"], 0)

    def test_evasive_dataset_degrades_gracefully(self):
        data = generate_dataset(evasive=True)
        agent = CaseAgent(data, CaseService(AuditLog()))
        m = evaluate(data.truth, {c: agent.investigate(ANALYST, c) for c in data.customers})
        self.assertLess(m["sar_recall"], 1.0)               # the rules really are evadable
        self.assertEqual(m["any_alert_recall"], 1.0)        # but nothing suspicious goes completely unreviewed
        self.assertEqual(m["sar_precision"], 1.0)

    def test_cases_cite_matching_policy_sections(self):
        structuring = next(r for c, r in self.results.items() if self.data.truth.get(c, {}).get("typology") == "structuring")
        self.assertIn("Structuring", [c["heading"] for c in structuring["citations"]])
        watch = next(r for c, r in self.results.items() if self.data.truth.get(c, {}).get("typology") == "watchlist_match")
        self.assertIn("Watchlist and Sanctions Matches", [c["heading"] for c in watch["citations"]])

    def test_every_reason_is_backed_by_transactions_or_profile(self):
        for r in self.results.values():
            for reason in r["risk"]["reasons"]:
                if reason["key"] in ("structuring", "rapid_movement", "velocity", "high_risk_geo", "large_cash"):
                    self.assertTrue(reason["tx_ids"], reason["key"])

    def test_no_case_for_clean_customers(self):
        clean = next(c for c in self.data.customers if c not in self.data.truth)
        self.assertIsNone(self.results[clean]["case_id"])
        self.assertEqual(self.results[clean]["recommendation"], "NO_ACTION")

    def test_every_tool_call_is_audited(self):
        actions = {e["action"] for e in self.audit.entries()}
        for tool in ("get_customer_profile", "kyc_checks", "screen_watchlist", "get_transactions",
                     "run_aml_rules", "lookup_policy"):
            self.assertIn(f"agent.{tool}", actions)
        self.assertTrue(self.audit.verify()[0])

    def test_auditor_cannot_run_agent_and_denial_is_logged(self):
        before = len(self.audit)
        with self.assertRaises(PermissionDenied):
            self.agent.investigate(AUDITOR, "C0001")
        last = self.audit.entries()[len(self.audit) - 1]
        self.assertEqual((last["outcome"], last["actor"]), ("denied", "meera"))
        self.assertGreater(len(self.audit), before)

    def test_llm_narrative_with_invented_numbers_is_rejected(self):
        cid = next(c for c, r in self.results.items() if r["recommendation"] == "ESCALATE_SAR_RECOMMENDED")
        liar = CaseAgent(self.data, CaseService(AuditLog()), narrative_writer=lambda facts: "Moved 999,999.99 to offshore accounts.")
        out = liar.investigate(ANALYST, cid)
        self.assertNotIn("999,999.99", out["narrative"])
        self.assertIn("risk score", out["narrative"])
        faithful = CaseAgent(self.data, CaseService(AuditLog()),
                             narrative_writer=lambda f: f"Risk score {f['assessment']['score']} for {f['customer_id']}.")
        self.assertTrue(faithful.investigate(ANALYST, cid)["narrative"].startswith("Risk score"))

    def test_verify_narrative(self):
        self.assertEqual(verify_narrative("Total 1,200.50 across 14 items", {"a": "1200.50", "b": 14}), [])
        self.assertEqual(verify_narrative("Total 5,000", {"a": 12}), ["5000"])


class CaseWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.data = generate_dataset()
        self.audit = AuditLog()
        self.service = CaseService(self.audit)
        self.agent = CaseAgent(self.data, self.service)
        cid = next(c for c, t in self.data.truth.items() if t["typology"] == "rapid_movement")
        self.case = self.agent.investigate(ANALYST, cid)["case_id"]

    def test_happy_path_with_four_eyes(self):
        self.service.act(ANALYST, self.case, "start_review")
        self.service.act(ANALYST, self.case, "escalate", "Funds pass straight through.")
        done = self.service.act(OFFICER, self.case, "approve_sar", "Evidence reviewed.")
        self.assertEqual(done["status"], "SAR_APPROVED")
        self.assertEqual([h["action"] for h in done["history"]], ["start_review", "escalate", "approve_sar"])

    def test_analyst_cannot_approve(self):
        self.service.act(ANALYST, self.case, "start_review")
        self.service.act(ANALYST, self.case, "escalate", "note")
        with self.assertRaises(PermissionDenied):
            self.service.act(ANALYST, self.case, "approve_sar", "self-approve")

    def test_preparer_cannot_approve_even_as_officer(self):
        self.service.act(ANALYST, self.case, "start_review")
        self.service.act(ANALYST, self.case, "escalate", "note")
        with self.assertRaises(WorkflowError):
            self.service.act(Principal("asha", "compliance_officer"), self.case, "approve_sar", "note")

    def test_cannot_skip_review(self):
        with self.assertRaises(WorkflowError):
            self.service.act(ANALYST, self.case, "escalate", "too early")

    def test_note_required(self):
        self.service.act(ANALYST, self.case, "start_review")
        with self.assertRaises(WorkflowError):
            self.service.act(ANALYST, self.case, "escalate", "  ")

    def test_closing_a_recommended_sar_needs_officer(self):
        self.service.act(ANALYST, self.case, "start_review")
        with self.assertRaises(PermissionDenied):
            self.service.act(SENIOR, self.case, "close_no_action", "looks fine")
        closed = self.service.act(OFFICER, self.case, "close_no_action", "Verified legitimate.")
        self.assertEqual(closed["status"], "CLOSED")

    def test_auditor_sees_masked_names_analyst_sees_full(self):
        name = self.data.customers[self.service.view_case(ANALYST, self.case)["customer_id"]].name
        self.assertEqual(self.service.view_case(ANALYST, self.case)["customer_name"], name)
        masked = self.service.view_case(AUDITOR, self.case)
        self.assertNotIn(name, json.dumps(masked))
        self.assertTrue(masked["customer_name"].startswith(name[0]))

    def test_admin_cannot_read_cases_but_can_read_audit(self):
        with self.assertRaises(PermissionDenied):
            self.service.view_case(ADMIN, self.case)
        self.assertEqual(self.audit.verify()[0], True)

    def test_failed_attempts_are_audited(self):
        with self.assertRaises(PermissionDenied):
            self.service.act(AUDITOR, self.case, "start_review")
        with self.assertRaises(WorkflowError):
            self.service.act(ANALYST, self.case, "escalate", "too early")
        outcomes = [e["outcome"] for e in self.audit.entries() if e["action"] in ("case.start_review", "case.escalate")]
        self.assertEqual(outcomes, ["denied", "rejected"])


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = generate_dataset()
        cls.agent = CaseAgent(data, CaseService(AuditLog()))
        cls.server = make_server(cls.agent, {"a": ("asha", "analyst"), "o": ("ravi", "compliance_officer"),
                                             "u": ("meera", "auditor"), "d": ("dev", "admin")})
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.sar_customer = next(c for c, t in data.truth.items() if t["typology"] == "structuring")

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def call(self, method, path, token=None, body=None):
        req = urllib.request.Request(self.base + path, method=method, data=json.dumps(body).encode() if body is not None else None)
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read())

    def test_auth_and_roles(self):
        self.assertEqual(self.call("GET", "/cases")[0], 401)
        self.assertEqual(self.call("GET", "/cases", "bogus")[0], 401)
        self.assertEqual(self.call("POST", "/investigate", "u", {"customer_id": self.sar_customer})[0], 403)
        self.assertEqual(self.call("GET", "/audit", "a")[0], 403)
        self.assertEqual(self.call("GET", "/cases", "d")[0], 403)

    def test_full_flow(self):
        status, result = self.call("POST", "/investigate", "a", {"customer_id": self.sar_customer})
        self.assertEqual(status, 200)
        case = result["case_id"]
        self.assertEqual(self.call("POST", f"/cases/{case}/start_review", "a")[0], 200)
        self.assertEqual(self.call("POST", f"/cases/{case}/escalate", "a", {"note": "structuring"})[0], 200)
        self.assertEqual(self.call("POST", f"/cases/{case}/approve_sar", "a", {"note": "x"})[0], 403)
        status, body = self.call("POST", f"/cases/{case}/approve_sar", "o", {"note": "approved"})
        self.assertEqual((status, body["status"]), (200, "SAR_APPROVED"))
        status, masked = self.call("GET", f"/cases/{case}", "u")
        self.assertEqual(status, 200)
        self.assertIn("*", masked["customer_name"])
        self.assertTrue(self.call("GET", "/audit/verify", "o")[1]["ok"])

    def test_errors(self):
        self.assertEqual(self.call("POST", "/investigate", "a", {"customer_id": "NOPE"})[0], 404)
        self.assertEqual(self.call("GET", "/cases/CASE-99999", "a")[0], 404)
        self.assertEqual(self.call("GET", "/nothing", "a")[0], 404)


if __name__ == "__main__":
    unittest.main()
