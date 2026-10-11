"""PolicyGuard: retrieval-grounded policy checker.

Checks a policy document against a requirements framework, flags vague wording,
placeholders and internal conflicts, and answers questions with citations.
"""

from .chunking import Clause, split_clauses
from .retrieval import Hit, HybridRetriever
from .checker import Finding, Report, check_policy, load_framework
from .qa import Answer, ask, grounding_check

__version__ = "0.1.0"
__all__ = [
    "Answer",
    "Clause",
    "Finding",
    "Hit",
    "HybridRetriever",
    "Report",
    "ask",
    "check_policy",
    "grounding_check",
    "load_framework",
    "split_clauses",
]
