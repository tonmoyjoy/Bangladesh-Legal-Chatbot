"""Citation checking: keep only ``[n]`` markers that point at a real source."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from legalbot.prompts import REFUSAL_BN, REFUSAL_EN
from legalbot.text import bn_to_ascii_digits

_BRACKET = re.compile(r"\[\s*([0-9০-৯]{1,2}(?:\s*[,،]\s*[0-9০-৯]{1,2})*)\s*\]")


@dataclass
class CitationResult:
    text: str
    cited: list[int] = field(default_factory=list)
    invalid: list[int] = field(default_factory=list)


def validate_citations(answer: str, n_sources: int) -> CitationResult:
    """Drop citation numbers outside ``1..n_sources`` and list the valid ones."""
    cited: set[int] = set()
    invalid: set[int] = set()

    def fix(match: re.Match[str]) -> str:
        numbers = [int(bn_to_ascii_digits(p.strip())) for p in re.split(r"[,،]", match.group(1))]
        good = [n for n in numbers if 1 <= n <= n_sources]
        invalid.update(n for n in numbers if not 1 <= n <= n_sources)
        cited.update(good)
        if not good:
            return ""
        return "".join(f"[{n}]" for n in good)

    text = _BRACKET.sub(fix, answer)
    text = re.sub(r"[ \t]+([.,;:।])", r"\1", text)  # tidy space left by removed markers
    text = re.sub(r"[ \t]{2,}", " ", text)
    return CitationResult(text=text.strip(), cited=sorted(cited), invalid=sorted(invalid))


def is_refusal(answer: str) -> bool:
    normalised = " ".join(answer.split())
    return REFUSAL_EN[:40] in normalised or REFUSAL_BN[:30] in normalised
