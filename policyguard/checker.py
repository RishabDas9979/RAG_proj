"""Check a policy document against a requirements framework."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import Clause, split_clauses
from .retrieval import HybridRetriever

FRAMEWORK_DIR = Path(__file__).parent / "frameworks"
DEFAULT_FRAMEWORK = "banking_baseline"
SEVERITY_WEIGHT = {"critical": 4, "high": 3, "medium": 2, "low": 1}
DISCLAIMER = (
    "Automated check against a generic baseline. It finds gaps in wording and structure; "
    "it is not legal or regulatory advice."
)

_NUM = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|twenty[- ]four|forty[- ]eight|seventy[- ]two)"
_TIME = re.compile(
    rf"\b{_NUM}\s*(?:\(\d+\))?\s*(?:calendar\s+|business\s+|working\s+)?(?:hours?|days?|weeks?|months?|years?)\b", re.I
)
_FREQ = re.compile(
    r"\b(annual(?:ly)?|quarterly|monthly|weekly|daily|bi-?annual(?:ly)?|semi-?annual(?:ly)?|"
    r"every\s+\w+\s+(?:days|weeks|months|years)|at least (?:once|twice)|once (?:a|per|every) (?:year|month|quarter)|"
    r"each (?:year|quarter|month)|continuous(?:ly)?|real[- ]time)\b",
    re.I,
)
_OWNER = re.compile(
    r"\b(officer|manager|head of|committee|ciso|cro|cco|mlro|director|department|team|board|responsible|owner|chief)\b",
    re.I,
)
_PLACEHOLDER = re.compile(r"\b(?:TBD|TODO|TBC|to be (?:decided|determined|defined|confirmed))\b|\[(?:insert|tbd|todo|xx+)[^\]]*\]", re.I)

_VAGUE_ALWAYS = [
    "as appropriate", "where appropriate", "where possible", "as needed", "as necessary", "when necessary",
    "reasonable efforts", "best efforts", "if feasible", "wherever practicable", "should consider", "etc.",
]
_VAGUE_UNLESS_QUANTIFIED = ["periodically", "regularly", "promptly", "in a timely manner", "from time to time", "soon as possible"]

_RETENTION_VERB = re.compile(r"\b(retain|retained|retention|kept|stored|preserved|maintained)\b", re.I)
_RETENTION_SUBJECTS = ["customer records", "transaction records", "kyc", "audit log", "logs", "records", "data"]
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class Citation:
    clause_id: str
    heading: str
    line: int
    snippet: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Finding:
    id: str
    kind: str          # requirement | vague_language | placeholder | conflict
    severity: str
    status: str        # met | partial | missing | issue
    title: str
    detail: str
    citations: list[Citation] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["citations"] = [c.to_dict() for c in self.citations]
        return d


@dataclass
class Report:
    framework: str
    score: float
    rating: str
    findings: list[Finding]
    clause_count: int
    disclaimer: str = DISCLAIMER

    @property
    def requirements(self) -> list[Finding]:
        return [f for f in self.findings if f.kind == "requirement"]

    @property
    def issues(self) -> list[Finding]:
        return [f for f in self.findings if f.kind != "requirement"]

    def counts(self) -> dict[str, int]:
        c = {"met": 0, "partial": 0, "missing": 0}
        for f in self.requirements:
            c[f.status] += 1
        return c

    def to_dict(self) -> dict:
        return {
            "framework": self.framework,
            "score": round(self.score, 3),
            "rating": self.rating,
            "counts": self.counts(),
            "clause_count": self.clause_count,
            "findings": [f.to_dict() for f in self.findings],
            "disclaimer": self.disclaimer,
        }

    def to_markdown(self) -> str:
        c = self.counts()
        lines = [
            f"# PolicyGuard report: {self.framework}",
            "",
            f"**Score: {self.score * 100:.0f}% ({self.rating})** | met {c['met']}, partial {c['partial']}, missing {c['missing']}"
            f" | {len(self.issues)} wording/consistency issue(s)",
            "",
            "## Requirements",
        ]
        for f in self.requirements:
            lines.append(f"- [{f.status.upper()}] {f.id} {f.title} ({f.severity}): {f.detail}")
            for cit in f.citations:
                lines.append(f"    - cites {cit.clause_id} \"{cit.heading}\" (line {cit.line})")
        if self.issues:
            lines += ["", "## Issues"]
            for f in self.issues:
                lines.append(f"- [{f.kind}] {f.title} ({f.severity}): {f.detail}")
                for cit in f.citations:
                    lines.append(f"    - {cit.clause_id} \"{cit.heading}\" (line {cit.line}): {cit.snippet}")
        lines += ["", f"_{self.disclaimer}_"]
        return "\n".join(lines)


def load_framework(name_or_path: str | Path = DEFAULT_FRAMEWORK) -> dict:
    p = Path(name_or_path)
    if not p.suffix:
        p = FRAMEWORK_DIR / f"{name_or_path}.json"
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _phrase_hits(text_lower: str, phrases: list[str]) -> list[str]:
    return [p for p in phrases if re.search(r"\b" + re.escape(p.lower()), text_lower)]


def _cite(clause: Clause, snippet: str | None = None) -> Citation:
    snippet = (snippet or clause.text).strip().replace("\n", " ")
    return Citation(clause.id, clause.heading, clause.start_line, snippet[:160])


def _vague_in(text: str) -> list[str]:
    found: list[str] = []
    for sentence in _sentences(text):
        low = sentence.lower()
        for phrase in _VAGUE_ALWAYS:
            if phrase in low and phrase not in found:
                found.append(phrase)
        for phrase in _VAGUE_UNLESS_QUANTIFIED:
            if phrase in low and not (_TIME.search(sentence) or _FREQ.search(sentence)) and phrase not in found:
                found.append(phrase)
    return found


def _days(value: str, unit: str) -> float | None:
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "ten": 10, "twelve": 12}
    n = float(value) if value.isdigit() else words.get(value.lower())
    if n is None:
        return None
    unit = unit.lower().rstrip("s")
    return n * {"year": 365, "month": 30, "week": 7, "day": 1, "hour": 1 / 24}.get(unit, 1)


def _retention_conflicts(clauses: list[Clause]) -> list[Finding]:
    seen: dict[str, list[tuple[float, str, Clause, str]]] = {}
    for clause in clauses:
        for sentence in _sentences(clause.text):
            if not _RETENTION_VERB.search(sentence):
                continue
            m = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve)\s*(?:\(\d+\))?\s*(years?|months?)\b", sentence, re.I)
            if not m:
                continue
            days = _days(m.group(1), m.group(2))
            if days is None:
                continue
            low = sentence.lower()
            subject = next((s for s in _RETENTION_SUBJECTS if s in low), None)
            if subject:
                seen.setdefault(subject, []).append((days, m.group(0), clause, sentence))
    findings: list[Finding] = []
    for subject, entries in seen.items():
        if len({round(e[0]) for e in entries}) > 1:
            cites = [_cite(e[2], e[3]) for e in entries[:4]]
            values = ", ".join(sorted({e[1] for e in entries}))
            findings.append(Finding(
                id=f"CONFLICT-{subject.upper().replace(' ', '-')}", kind="conflict", severity="high", status="issue",
                title=f"Conflicting retention periods for {subject}",
                detail=f"The policy states different periods ({values}) for the same subject.", citations=cites,
            ))
    return findings


def check_policy(text: str, framework: dict | str | Path | None = None, top_evidence: int = 3) -> Report:
    fw = framework if isinstance(framework, dict) else load_framework(framework or DEFAULT_FRAMEWORK)
    clauses = split_clauses(text)
    retriever = HybridRetriever(clauses)
    findings: list[Finding] = []

    for req in fw["requirements"]:
        any_of = req["any_of"]
        evidence = [c for c in clauses if _phrase_hits(f"{c.heading} {c.text}".lower(), any_of)]
        if evidence:
            rank = {h.clause.id: i for i, h in enumerate(retriever.search(req["query"], k=len(clauses)))}
            evidence.sort(key=lambda c: rank.get(c.id, 10**6))
            evidence = evidence[:top_evidence]
            blob = " ".join(c.text for c in evidence)
            gaps: list[str] = []
            if req.get("needs_owner") and not _OWNER.search(blob):
                gaps.append("no accountable owner or role named")
            if req.get("needs_frequency") and not _FREQ.search(blob):
                gaps.append("no frequency stated")
            if req.get("needs_timebound") and not _TIME.search(blob):
                gaps.append("no time limit stated")
            for exp in req.get("also_expect", []):
                if not _phrase_hits(blob.lower(), exp["any_of"]):
                    gaps.append(f"does not cover {exp['label']}")
            vague = _vague_in(blob)
            if vague:
                gaps.append("vague wording (" + ", ".join(f"'{v}'" for v in vague) + ")")
            status = "partial" if gaps else "met"
            detail = ("Gaps: " + "; ".join(gaps) + ".") if gaps else "Requirement is addressed with the expected specifics."
            findings.append(Finding(req["id"], "requirement", req["severity"], status, req["title"], detail,
                                    [_cite(c) for c in evidence]))
        else:
            near = [h for h in retriever.search(req["query"], k=1) if h.coverage >= 0.4]
            hint = f" Closest section: {near[0].clause.id} \"{near[0].clause.heading}\"." if near else ""
            findings.append(Finding(req["id"], "requirement", req["severity"], "missing", req["title"],
                                    "Not found in the policy." + hint, [_cite(near[0].clause)] if near else []))

    total = sum(SEVERITY_WEIGHT[f.severity] for f in findings)
    earned = sum(SEVERITY_WEIGHT[f.severity] * {"met": 1.0, "partial": 0.5, "missing": 0.0}[f.status] for f in findings)
    score = earned / total if total else 0.0

    # Cross-cutting issues
    vague_cites: list[Citation] = []
    vague_terms: list[str] = []
    for clause in clauses:
        terms = _vague_in(clause.text)
        if terms:
            vague_cites.append(_cite(clause))
            vague_terms += [t for t in terms if t not in vague_terms]
    if vague_cites:
        findings.append(Finding("VAGUE-01", "vague_language", "medium", "issue", "Vague or unenforceable wording",
                                "Replace with owners, numbers and dates: " + ", ".join(f"'{t}'" for t in vague_terms) + ".",
                                vague_cites[:5]))
    placeholders = []
    for clause in clauses:
        for sentence in _sentences(clause.text):
            if _PLACEHOLDER.search(sentence):
                placeholders.append(_cite(clause, sentence))
    if placeholders:
        findings.append(Finding("PLACEHOLDER-01", "placeholder", "high", "issue", "Unfinished placeholder text",
                                f"{len(placeholders)} placeholder(s) such as TBD/TODO remain.", placeholders[:5]))
    findings.extend(_retention_conflicts(clauses))

    penalty = 0.03 * sum(1 for f in findings if f.kind != "requirement")
    score = max(0.0, score - min(penalty, 0.15))
    rating = "Strong" if score >= 0.85 else "Needs work" if score >= 0.6 else "Weak"
    return Report(fw["name"], score, rating, findings, len(clauses))
