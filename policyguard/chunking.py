"""Structure-aware chunking: split a policy into clauses along its headings."""

from __future__ import annotations

import re
from dataclasses import dataclass

_MD_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_DECIMAL_HEADING = re.compile(r"^\s{0,3}(\d+\.\d+(?:\.\d+)*)[.)]?\s+([A-Z][^.!?]{1,90})$")
_SECTION_HEADING = re.compile(r"^\s{0,3}(?:Section|Article|Clause)\s+(\d+)[.:]?\s+(.{1,90})$", re.I)


@dataclass(frozen=True)
class Clause:
    id: str
    heading: str
    text: str
    start_line: int


def _parse_heading(line: str):
    """Return (heading_text, number_or_None) when the line is a heading."""
    m = _MD_HEADING.match(line)
    if m:
        title = m.group(2).strip()
        numbered = re.match(r"^(\d+(?:\.\d+)*)[.)]?\s+(\S.*)$", title)
        if numbered:
            return numbered.group(2).strip(), numbered.group(1)
        return title, None
    m = _DECIMAL_HEADING.match(line)
    if m:
        return m.group(2).strip(), m.group(1)
    m = _SECTION_HEADING.match(line)
    if m:
        return m.group(2).strip(), m.group(1)
    return None


def split_clauses(text: str, max_chars: int = 1200) -> list[Clause]:
    """Split ``text`` into clauses at headings; long sections are split by paragraph."""
    sections: list[tuple[str, str | None, int, list[str]]] = []
    current: tuple[str, str | None, int, list[str]] = ("Preamble", None, 1, [])
    for line_no, line in enumerate(text.splitlines(), 1):
        heading = _parse_heading(line)
        if heading:
            if "".join(current[3]).strip():
                sections.append(current)
            current = (heading[0], heading[1], line_no, [])
        else:
            current[3].append(line)
    if "".join(current[3]).strip():
        sections.append(current)

    clauses: list[Clause] = []
    seen: set[str] = set()
    for n, (heading, number, start, body) in enumerate(sections, 1):
        base = number or f"S{n}"
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", "\n".join(body)) if p.strip()]
        pieces: list[str] = []
        buf = ""
        for para in paragraphs:
            if buf and len(buf) + len(para) + 2 > max_chars:
                pieces.append(buf)
                buf = para
            else:
                buf = f"{buf}\n\n{para}" if buf else para
        if buf:
            pieces.append(buf)
        for j, piece in enumerate(pieces, 1):
            cid = base if len(pieces) == 1 else f"{base}#{j}"
            while cid in seen:
                cid += "'"
            seen.add(cid)
            clauses.append(Clause(cid, heading, piece, start))
    return clauses
