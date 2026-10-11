import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from policyguard import HybridRetriever, ask, check_policy, grounding_check, split_clauses
from policyguard.cli import main as cli_main

ROOT = Path(__file__).resolve().parent.parent
GOOD = (ROOT / "data/policies/good_policy.md").read_text(encoding="utf-8")
WEAK = (ROOT / "data/policies/weak_policy.md").read_text(encoding="utf-8")


class ChunkingTests(unittest.TestCase):
    def test_splits_on_headings(self):
        clauses = split_clauses(GOOD)
        self.assertIn("Encryption", [c.heading for c in clauses])
        self.assertGreaterEqual(len(clauses), 16)

    def test_section_numbers_become_ids(self):
        by_heading = {c.heading: c.id for c in split_clauses(GOOD)}
        self.assertEqual(by_heading["Incident Response"], "8")

    def test_ids_unique_and_lines_positive(self):
        clauses = split_clauses(GOOD)
        self.assertEqual(len({c.id for c in clauses}), len(clauses))
        self.assertTrue(all(c.start_line > 0 for c in clauses))

    def test_long_section_is_split(self):
        text = "## Big\n\n" + "\n\n".join(f"Paragraph {i} " + "word " * 80 for i in range(10))
        clauses = split_clauses(text, max_chars=500)
        self.assertGreater(len(clauses), 1)
        self.assertTrue(all("#" in c.id for c in clauses))

    def test_empty_text(self):
        self.assertEqual(split_clauses(""), [])


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.retriever = HybridRetriever(split_clauses(GOOD))

    def test_relevant_clause_ranks_first(self):
        top = self.retriever.search("how long are customer records retained", k=3)[0]
        self.assertIn("Retention", top.clause.heading)

    def test_incident_query(self):
        top = self.retriever.search("breach notification to the regulator", k=1)[0]
        self.assertIn("Incident", top.clause.heading)

    def test_unrelated_query_returns_nothing_or_low_coverage(self):
        hits = self.retriever.search("quantum chromodynamics lagrangian", k=3)
        self.assertTrue(not hits or hits[0].coverage < 0.34)

    def test_empty_inputs(self):
        self.assertEqual(self.retriever.search("", k=3), [])
        self.assertEqual(HybridRetriever([]).search("anything"), [])


class CheckerTests(unittest.TestCase):
    def test_good_policy_scores_strong(self):
        report = check_policy(GOOD)
        self.assertGreaterEqual(report.score, 0.85, report.to_markdown())
        self.assertEqual(report.counts()["missing"], 0, report.to_markdown())
        self.assertEqual(report.rating, "Strong")

    def test_weak_policy_scores_weak_and_is_lower(self):
        good, weak = check_policy(GOOD), check_policy(WEAK)
        self.assertLess(weak.score, 0.5, weak.to_markdown())
        self.assertLess(weak.score, good.score)

    def test_weak_policy_flags_missing_controls(self):
        weak = check_policy(WEAK)
        status = {f.id: f.status for f in weak.requirements}
        for req_id in ("ACC-03", "FIN-03", "BCP-01", "VEN-01"):
            self.assertEqual(status[req_id], "missing", req_id)

    def test_weak_policy_flags_conflict_placeholder_and_vagueness(self):
        weak = check_policy(WEAK)
        kinds = {f.kind for f in weak.issues}
        self.assertEqual(kinds, {"conflict", "placeholder", "vague_language"})
        conflict = next(f for f in weak.issues if f.kind == "conflict")
        self.assertGreaterEqual(len(conflict.citations), 2)

    def test_good_policy_has_no_conflicts_or_placeholders(self):
        kinds = {f.kind for f in check_policy(GOOD).issues}
        self.assertFalse(kinds & {"conflict", "placeholder"})

    def test_partial_when_gaps(self):
        text = "## Sanctions\nThe bank screens against sanctions lists."
        report = check_policy(text)
        fin03 = next(f for f in report.requirements if f.id == "FIN-03")
        self.assertEqual(fin03.status, "partial")
        self.assertIn("no frequency", fin03.detail)

    def test_every_met_requirement_has_citation(self):
        for f in check_policy(GOOD).requirements:
            if f.status == "met":
                self.assertTrue(f.citations, f.id)

    def test_empty_policy_scores_zero(self):
        report = check_policy("")
        self.assertEqual(report.score, 0.0)
        self.assertEqual(report.counts()["met"], 0)

    def test_report_serialises(self):
        d = check_policy(WEAK).to_dict()
        json.dumps(d)
        self.assertIn("disclaimer", d)


class QATests(unittest.TestCase):
    def test_answer_with_citation(self):
        a = ask(GOOD, "How quickly must breach notification be made to the regulator?")
        self.assertTrue(a.confident)
        self.assertIn("72 hours", a.answer)
        self.assertEqual((a.citations[0].clause_id, a.citations[0].heading), ("8", "Incident Response"))

    def test_abstains_when_irrelevant(self):
        a = ask(GOOD, "What is the cafeteria menu on Fridays?")
        self.assertFalse(a.confident)
        self.assertEqual(a.citations, [])

    def test_grounding_flags_unsupported_sentence(self):
        clauses = split_clauses(GOOD)
        answer = "Customer records are retained for 7 years after the account closes. The bank also stores backups on the moon."
        checks = grounding_check(answer, clauses)
        self.assertTrue(checks[0]["supported"])
        self.assertFalse(checks[1]["supported"])

    def test_llm_hook_is_grounding_checked(self):
        ok = ask(GOOD, "How long are audit logs retained?", llm=lambda p: "Audit logs are retained for 7 years.")
        self.assertTrue(ok.confident)
        bad = ask(GOOD, "How long are audit logs retained?",
                  llm=lambda p: "Audit logs are retained forever on a blockchain run by pirates.")
        self.assertFalse(bad.confident)


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli_main(list(args))
        return code, buf.getvalue()

    def test_check_json_and_fail_under(self):
        code, out = self.run_cli("check", str(ROOT / "data/policies/weak_policy.md"), "--format", "json")
        self.assertEqual(code, 0)
        self.assertIn("findings", json.loads(out))
        code, _ = self.run_cli("check", str(ROOT / "data/policies/weak_policy.md"), "--fail-under", "80")
        self.assertEqual(code, 1)
        code, _ = self.run_cli("check", str(ROOT / "data/policies/good_policy.md"), "--fail-under", "80")
        self.assertEqual(code, 0)

    def test_ask(self):
        code, out = self.run_cli("ask", str(ROOT / "data/policies/good_policy.md"), "How often is MFA required?")
        self.assertIn("source", out)


if __name__ == "__main__":
    unittest.main()
