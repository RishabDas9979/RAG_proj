"""Question answering over a policy, with citations and a grounding guard."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .checker import Citation, _cite, _sentences
from .chunking import Clause, split_clauses
from .retrieval import Hit, HybridRetriever, query_tokens, tokenize

DISCLAIMER = "Answers are extracted from the supplied policy text only. This is not legal advice."


@dataclass
class Answer:
    question: str
    answer: str
    confident: bool
    citations: list[Citation] = field(default_factory=list)
    grounding: list[dict] | None = None
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "confident": self.confident,
            "citations": [c.to_dict() for c in self.citations],
            "grounding": self.grounding,
            "disclaimer": self.disclaimer,
        }


def grounding_check(answer: str, hits: list[Hit] | list[Clause], threshold: float = 0.6) -> list[dict]:
    """Check each sentence of ``answer`` against the retrieved clauses.

    A sentence is *supported* when at least ``threshold`` of its content words appear in one clause.
    This is a lexical guard against unsupported statements, not a semantic entailment check.
    """
    clauses = [h.clause if isinstance(h, Hit) else h for h in hits]
    clause_tokens = [(c, set(tokenize(c.text))) for c in clauses]
    results = []
    for sentence in _sentences(answer):
        tokens = set(tokenize(sentence))
        if not tokens:
            continue
        best_overlap, best_clause = 0.0, None
        for clause, ctoks in clause_tokens:
            overlap = len(tokens & ctoks) / len(tokens)
            if overlap > best_overlap:
                best_overlap, best_clause = overlap, clause
        results.append({
            "sentence": sentence,
            "supported": best_overlap >= threshold,
            "overlap": round(best_overlap, 2),
            "clause_id": best_clause.id if best_clause else None,
        })
    return results


def _best_sentences(question: str, hits: list[Hit], limit: int = 2) -> list[tuple[str, Clause]]:
    q = set(query_tokens(question))
    scored = []
    for hit in hits:
        for sentence in _sentences(hit.clause.text):
            toks = set(tokenize(sentence))
            if not toks:
                continue
            overlap = len(q & toks)
            if overlap:
                scored.append((overlap / (len(toks) ** 0.5), sentence, hit.clause))
    scored.sort(key=lambda s: -s[0])
    return [(s, c) for _, s, c in scored[:limit]]


def ask(
    policy: str | list[Clause],
    question: str,
    k: int = 3,
    min_coverage: float = 0.34,
    llm: Callable[[str], str] | None = None,
) -> Answer:
    """Answer ``question`` from ``policy``. Abstains when nothing relevant is found.

    Without ``llm`` the answer is extractive (the best sentences, quoted from the policy).
    With ``llm`` (a callable taking a prompt and returning text) the generated answer is checked
    sentence-by-sentence against the retrieved clauses and unsupported sentences are flagged.
    """
    clauses = split_clauses(policy) if isinstance(policy, str) else policy
    hits = HybridRetriever(clauses).search(question, k=k)
    if not hits or hits[0].coverage < min_coverage:
        return Answer(question, "No sufficiently relevant text was found in the policy.", False)

    citations = [_cite(h.clause) for h in hits]
    if llm is None:
        picks = _best_sentences(question, hits)
        if not picks:
            return Answer(question, "No sufficiently relevant text was found in the policy.", False)
        text = " ".join(s for s, _ in picks)
        cites = []
        for _, clause in picks:
            if clause.id not in {c.clause_id for c in cites}:
                cites.append(_cite(clause))
        return Answer(question, text, True, cites)

    context = "\n\n".join(f"[{h.clause.id}] {h.clause.heading}\n{h.clause.text}" for h in hits)
    prompt = (
        "Answer using only the policy excerpts below. If they do not answer the question, say so.\n\n"
        f"{context}\n\nQuestion: {question}"
    )
    generated = llm(prompt)
    checks = grounding_check(generated, hits)
    return Answer(question, generated, all(c["supported"] for c in checks) if checks else False, citations, checks)
