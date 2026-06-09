"""
actions.py — Action item extraction from meeting transcripts.

Uses regex heuristics to find commitments, todos, and follow-ups without
requiring a network call or LLM. Fast and reliable for common patterns.

Patterns detected:
  - "I will / we will / [name] will ..."
  - "action item: ..."
  - "follow up on ..."
  - "TODO: ..." / "to-do: ..."
  - "need to ..." / "needs to ..."
  - "by [date/day] ..."
  - "let's make sure ..."
  - "don't forget to ..."
"""

from __future__ import annotations

import re
from pathlib import Path

# Patterns that signal an action item. Each is case-insensitive.
_PATTERNS = [
    r"\b(?:I|we|you|[A-Z][a-z]+)\s+will\s+\w",
    r"\baction\s+item\s*[:\-]\s*.+",
    r"\bfollow[\s\-]?up\s+(?:on|with|about)\b.+",
    r"\b(?:TODO|to[\s\-]?do)\s*[:\-].+",
    r"\b(?:need|needs)\s+to\s+\w",
    r"\blet'?s\s+make\s+sure\b.+",
    r"\bdon'?t\s+forget\s+to\b.+",
    r"\bremember\s+to\b.+",
    r"\bdeadline\s+(?:is|:).+",
    r"\bby\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|EOD|end\s+of\s+(?:day|week|month)|next\s+week)\b.+",
    r"\bassign(?:ed)?\s+to\b.+",
    r"\bsend\s+(?:the|a|an)\b.+",
    r"\bschedule\b.+meeting.+",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]


def extract_action_items(transcript: str) -> list[str]:
    """
    Return a deduplicated list of sentences that contain action item signals.
    Splits transcript into sentences, applies pattern matching, returns matches.
    """
    # Simple sentence split on period/question/exclamation followed by space+capital
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', transcript)

    seen: set[str] = set()
    items: list[str] = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 10:
            continue
        for pattern in _COMPILED:
            if pattern.search(sentence):
                normalized = sentence.rstrip(".!?")
                if normalized not in seen:
                    seen.add(normalized)
                    items.append(normalized)
                break  # one match per sentence is enough

    return items


def write_action_items(audio_path: Path, items: list[str]) -> Path | None:
    """Write action items to a file alongside the transcript. Returns path or None if empty."""
    if not items:
        return None

    stem = audio_path.stem
    out_path = audio_path.parent / f"{stem}_action_items.txt"

    lines = [f"• {item}" for item in items]
    out_path.write_text(
        f"Fly on the Wall — Action Items\n"
        f"{'=' * 50}\n\n"
        + "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )
    return out_path
