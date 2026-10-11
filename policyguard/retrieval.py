"""Hybrid retrieval: BM25 + TF-IDF cosine, fused with reciprocal rank fusion."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .chunking import Clause

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    """a an and are as at be been by for from has have how in into is it its of on or
    that the their there these this to was were what when where which who whom will with
    shall does do did can could should would about any all each per than then also not
    policy often many much""".split()
)
_SUFFIXES = ("ations", "ation", "ings", "ing", "ions", "ion", "ies", "ed", "es", "s")


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP]


# Banking/compliance acronyms, expanded on the query side so "MFA" finds "multi-factor authentication".
ACRONYMS = {
    "mfa": "multi factor authentication", "2fa": "two factor authentication",
    "kyc": "know your customer", "cdd": "customer due diligence", "edd": "enhanced due diligence",
    "pep": "politically exposed person", "aml": "anti money laundering",
    "sar": "suspicious activity report", "str": "suspicious transaction report",
    "bcp": "business continuity plan", "dr": "disaster recovery", "rto": "recovery time objective",
    "dpo": "data protection officer", "ciso": "chief information security officer",
    "mlro": "money laundering reporting officer",
}


def query_units(query: str) -> list[list[set[str]]]:
    """One unit per query word; each unit lists alternative token sets that satisfy it."""
    units: list[list[set[str]]] = []
    for raw in _TOKEN.findall(query.lower()):
        if raw in _STOP:
            continue
        stem = _stem(raw)
        alternatives = [{stem}]
        expansion = ACRONYMS.get(raw) or ACRONYMS.get(stem)
        if expansion:
            alternatives.append(set(tokenize(expansion)))
        units.append(alternatives)
    return units


def query_tokens(query: str) -> list[str]:
    """Flattened scoring tokens for a query, including acronym expansions."""
    out: list[str] = []
    for unit in query_units(query):
        for alt in unit:
            out.extend(sorted(alt))
    return out


def _analyze(text: str) -> list[str]:
    tokens = tokenize(text)
    return tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.n = len(docs)
        self.avgdl = (sum(map(len, docs)) / self.n) if self.n else 0.0
        self.tf = [Counter(d) for d in docs]
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = [0.0] * self.n
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            norm = 1 - self.b + self.b * dl / (self.avgdl or 1.0)
            for term in set(query):
                f = tf.get(term, 0)
                if f:
                    out[i] += self.idf[term] * f * (self.k1 + 1) / (f + self.k1 * norm)
        return out


@dataclass(frozen=True)
class Hit:
    clause: Clause
    score: float      # fused reciprocal-rank score
    bm25: float
    cosine: float
    coverage: float   # share of query terms that appear in the clause


class HybridRetriever:
    def __init__(self, clauses: list[Clause], rrf_k: int = 60):
        self.clauses = list(clauses)
        self.rrf_k = rrf_k
        self._tokens = [set(tokenize(f"{c.heading} {c.text}")) for c in self.clauses]
        docs = [tokenize(f"{c.heading} {c.text}") for c in self.clauses]
        self._bm25 = BM25(docs)
        self._vec = None
        if self.clauses and any(docs):
            self._vec = TfidfVectorizer(analyzer=_analyze, sublinear_tf=True)
            self._matrix = self._vec.fit_transform([f"{c.heading} {c.text}" for c in self.clauses])

    def search(self, query: str, k: int = 5) -> list[Hit]:
        units = query_units(query)
        q_tokens = query_tokens(query)
        if not q_tokens or not self.clauses:
            return []
        bm25 = self._bm25.scores(q_tokens)
        cosine = [0.0] * len(self.clauses)
        if self._vec is not None:
            cosine = list(linear_kernel(self._vec.transform([query]), self._matrix)[0])

        fused = [0.0] * len(self.clauses)
        for scores in (bm25, cosine):
            ranked = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: -scores[i])
            for rank, i in enumerate(ranked, 1):
                fused[i] += 1.0 / (self.rrf_k + rank)

        def coverage(i: int) -> float:
            covered = sum(1 for unit in units if any(alt <= self._tokens[i] for alt in unit))
            return covered / len(units)

        order = sorted((i for i, s in enumerate(fused) if s > 0), key=lambda i: (-fused[i], i))
        return [
            Hit(clause=self.clauses[i], score=fused[i], bm25=bm25[i], cosine=cosine[i], coverage=coverage(i))
            for i in order[:k]
        ]
