"""Indian vehicle registration-number logic (pure functions: no OCR, Streamlit or OpenCV imports).

What this module does
---------------------
* Describes the common Indian registration formats as *slot templates* (letter / digit / literal).
* Scores how strongly a string looks like a registration (``validate``).
* Proposes a character-confusion correction ONLY when it is position-justified, unique, within a
  small budget, and ends in a known state/UT code (``propose_correction``). Otherwise it reports why
  no correction was made, and the caller keeps the raw text and marks the result uncertain.
* Recognises incidental plate markings (``IND`` ...) by vocabulary.

What it deliberately does NOT do
--------------------------------
* It never inserts or deletes characters and never forces a string to match a pattern.
* It does not claim that a format-valid string is the *correct* plate: validity is evidence,
  not proof.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# State / UT RTO prefixes (current plus common legacy codes). An unknown prefix lowers the evidence
# score but never makes a string invalid: new codes are issued over time.
STATE_CODES = frozenset(
    "AN AP AR AS BR CG CH CT DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP MZ NL OD OR PB PY "
    "RJ SK TN TR TS UA UK UP WB".split()
)

# Glyph-confusion tables. A character is only ever converted into the class its position requires.
DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"}
LETTER_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "C": "0", "I": "1", "Z": "2", "A": "4",
                   "S": "5", "G": "6", "B": "8"}

# Plate markings that are not part of the registration number.
INCIDENTAL_WORDS = ("INDIA", "IND")  # longest first


def clean(text: str) -> str:
    """Upper-case and keep only A-Z / 0-9."""
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


@dataclass(frozen=True)
class Template:
    name: str
    tier: int                   # 2 = modern/common, 1 = legacy or unusual (weaker evidence)
    slots: tuple[str, ...]      # "L", "D", or "=X" for a literal character X
    correctable: bool = True
    has_state: bool = True      # first two characters are a state/UT code


def _build_templates() -> dict[int, list[Template]]:
    by_len: dict[int, list[Template]] = {}

    def add(t: Template) -> None:
        by_len.setdefault(len(t.slots), []).append(t)

    for d in (1, 2):
        for a in (1, 2, 3):
            add(Template("state_standard", 2, ("L",) * 2 + ("D",) * d + ("L",) * a + ("D",) * 4))
            for s in (1, 2, 3):  # older plates with unpadded serial numbers
                add(Template("state_short_serial", 1, ("L",) * 2 + ("D",) * d + ("L",) * a + ("D",) * s))
    for a in (1, 2):  # Bharat (BH) series: YY BH #### XX
        add(Template("bharat_series", 2, ("D", "D", "=B", "=H") + ("D",) * 4 + ("L",) * a,
                     has_state=False))
    for k in (1, 2, 3):  # diplomatic: digits + CD/CC/UN + digits (never auto-corrected)
        for lit in ("CD", "CC", "UN"):
            for m in (1, 2, 3, 4):
                add(Template("diplomatic", 1, ("D",) * k + tuple("=" + c for c in lit) + ("D",) * m,
                             correctable=False, has_state=False))
    return by_len


TEMPLATES = _build_templates()


def _slot_ok(ch: str, slot: str) -> bool:
    if slot == "L":
        return "A" <= ch <= "Z"
    if slot == "D":
        return "0" <= ch <= "9"
    return ch == slot[1:]


@dataclass(frozen=True)
class Evidence:
    valid: bool
    format: str = ""
    tier: int = 0
    state_code: str = ""
    state_known: bool | None = None  # None when the format has no state prefix
    note: str = ""


_DISTRICT = re.compile(r"^[A-Z]{2}(\d+)")


def _semantic_ok(text: str, tpl: Template) -> bool:
    """Rules a slot pattern cannot express: an RTO/district number is never zero."""
    if tpl.name.startswith("state_"):
        m = _DISTRICT.match(text)
        return bool(m) and int(m.group(1)) > 0
    return True


def validate(text: str) -> Evidence:
    """Does ``text`` (already cleaned) match a supported registration format?"""
    text = clean(text)
    if not text:
        return Evidence(False, note="empty")
    best: Evidence | None = None
    for tpl in TEMPLATES.get(len(text), ()):
        if all(_slot_ok(c, s) for c, s in zip(text, tpl.slots)) and _semantic_ok(text, tpl):
            state = text[:2] if tpl.has_state else ""
            ev = Evidence(True, tpl.name, tpl.tier, state,
                          (state in STATE_CODES) if tpl.has_state else None)
            if best is None or ev.tier > best.tier:
                best = ev
    if best:
        return best
    return Evidence(False, note=f"no supported format has {len(text)} characters in this letter/digit pattern")


@dataclass(frozen=True)
class Correction:
    text: str
    changes: tuple[tuple[int, str, str], ...]   # (1-based position, from, to)
    format: str

    def describe(self) -> list[str]:
        return [f"{a}->{b}@{p}" for p, a, b in self.changes]


def _fit(text: str, tpl: Template) -> list[tuple[int, str, str]] | None:
    changes: list[tuple[int, str, str]] = []
    for i, (ch, slot) in enumerate(zip(text, tpl.slots)):
        if _slot_ok(ch, slot):
            continue
        if slot == "L":
            repl = DIGIT_TO_LETTER.get(ch)
        elif slot == "D":
            repl = LETTER_TO_DIGIT.get(ch)
        else:
            repl = None  # a literal can never be "converted"
        if repl is None:
            return None
        changes.append((i + 1, ch, repl))
    return changes


def _single_deletions_valid(text: str) -> set[str]:
    """Modern-format valid strings reachable by deleting ONE character (used only to detect ambiguity)."""
    out = set()
    for i in range(len(text)):
        cand = text[:i] + text[i + 1:]
        ev = validate(cand)
        # only a modern-format result is a credible rival; legacy short-serial forms are too easy to hit by chance
        if ev.valid and ev.tier == 2 and ev.state_known is not False:
            out.add(cand)
    return out


def propose_correction(text: str, max_changes: int = 2) -> tuple[Correction | None, str]:
    """Return (correction, reason). ``correction`` is None when none is justified.

    A correction is proposed only if ALL hold:
      1. the text is not already a valid registration;
      2. some template of the same length fits after converting at most ``max_changes`` characters,
         each into the class its position requires (no insertions, no deletions);
      3. the cheapest fix is unique (ties between different strings = ambiguous = no correction);
      4. the corrected string has a known state/UT code;
      5. no single-character *deletion* would also yield a valid registration (a different repair
         hypothesis means the substitution is not the only explanation).
    """
    text = clean(text)
    if not text:
        return None, "empty text"
    if validate(text).valid:
        return None, "already a valid registration format"
    fits: dict[str, tuple[Correction, int]] = {}
    for tpl in TEMPLATES.get(len(text), ()):
        if not tpl.correctable:
            continue
        ch = _fit(text, tpl)
        if ch is None or not ch or len(ch) > max_changes:
            continue
        chars = list(text)
        for pos, _, to in ch:
            chars[pos - 1] = to
        new = "".join(chars)
        if tpl.has_state and new[:2] not in STATE_CODES:
            continue
        if not _semantic_ok(new, tpl):
            continue
        cur = fits.get(new)
        if cur is None or len(ch) < cur[1]:
            fits[new] = (Correction(new, tuple(ch), tpl.name), len(ch))
    if not fits:
        n = len(text)
        if n not in TEMPLATES:
            return None, f"{n} characters does not fit any supported format (missing or extra characters)"
        return None, f"no position-justified fix within {max_changes} character conversions"
    cost = min(c for _, c in fits.values())
    cheapest = [corr for corr, c in fits.values() if c == cost]
    if len(cheapest) > 1:
        return None, "ambiguous: several different corrections are equally plausible"
    chosen = cheapest[0]
    rivals = _single_deletions_valid(text) - {chosen.text}
    if rivals:
        return None, "ambiguous: removing one character would also give a valid registration"
    return chosen, "position-justified correction"


def split_incidental(token: str) -> tuple[str, str] | None:
    """If ``token`` is/starts with/ends with an incidental word, return (word, remainder)."""
    token = clean(token)
    for w in INCIDENTAL_WORDS:
        if token == w:
            return w, ""
        if len(token) > len(w) and (token.startswith(w) or token.endswith(w)):
            rest = token[len(w):] if token.startswith(w) else token[:-len(w)]
            return w, rest
    return None