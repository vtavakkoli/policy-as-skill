from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable

from .utils import tokens

DECISIONS = {"allowed", "not_allowed", "conditional", "needs_review", "unknown"}

_MODAL_PATTERNS = {
    "prohibited": re.compile(r"\b(must not|shall not|may not|is prohibited|are prohibited|not allowed|forbidden)\b", re.I),
    "mandatory": re.compile(r"\b(must|shall|required|required to|requires|require|only if|before use|prior to)\b", re.I),
    "permitted": re.compile(r"\b(may|is allowed|are allowed|permitted|can proceed)\b", re.I),
    "advisory": re.compile(r"\b(should|recommended|preferably|good practice)\b", re.I),
}
_MISSING_PATTERN = re.compile(
    r"\b(without|missing|omits?|lacks?|absence of|not (?:recorded|provided|performed|completed|available|documented))\b",
    re.I,
)
_REVIEW_PATTERN = re.compile(r"\b(human review|manual review|oversight|approval|escalat\w*|policy owner)\b", re.I)
_CONFLICT_PATTERN = re.compile(r"\b(conflict|contradict\w*|inconsistent|precedence|stricter|supersed\w*)\b", re.I)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "before", "by", "can", "does", "for", "from",
    "how", "if", "in", "is", "it", "may", "must", "of", "on", "or", "should", "the", "to", "what",
    "when", "where", "which", "with", "without", "would",
}


@dataclass(frozen=True)
class PolicyRequirement:
    citation_id: str
    text: str
    modality: str
    tags: tuple[str, ...]
    overlap: float


@dataclass(frozen=True)
class ControlAssessment:
    decision: str
    human_review_required: bool
    confidence: float
    rationale: str
    matched_requirements: tuple[PolicyRequirement, ...]
    missing_mandatory_control: bool
    conflict_detected: bool

    def to_dict(self) -> dict:
        out = asdict(self)
        out["matched_requirements"] = [asdict(x) for x in self.matched_requirements]
        return out


def _content_tokens(text: str) -> set[str]:
    return {t for t in tokens(text) if t not in _STOPWORDS and len(t) > 2}


def _sentence_split(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if not clean:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+", clean) if s.strip()]


def _modality(sentence: str) -> str:
    for name in ("prohibited", "mandatory", "permitted", "advisory"):
        if _MODAL_PATTERNS[name].search(sentence):
            return name
    return "descriptive"


def _overlap(question: str, sentence: str, tags: Iterable[str] = ()) -> float:
    q = _content_tokens(question)
    s = _content_tokens(sentence) | {t.lower() for tag in tags for t in re.split(r"[-_\s]+", tag) if t}
    if not q or not s:
        return 0.0
    return len(q & s) / max(1, len(q))


def extract_requirements(question: str, evidence: list[dict], min_overlap: float = 0.08) -> list[PolicyRequirement]:
    """Extract applicable normative clauses without reading benchmark labels."""
    out: list[PolicyRequirement] = []
    for item in evidence or []:
        citation_id = str(item.get("citation_id", ""))
        tags = tuple(str(x) for x in item.get("tags", []) or [])
        for sentence in _sentence_split(str(item.get("text", ""))):
            score = _overlap(question, sentence, tags)
            if score < min_overlap:
                continue
            out.append(
                PolicyRequirement(
                    citation_id=citation_id,
                    text=sentence,
                    modality=_modality(sentence),
                    tags=tags,
                    overlap=score,
                )
            )
    return sorted(out, key=lambda r: (r.overlap, r.modality in {"prohibited", "mandatory"}), reverse=True)


def assess_policy(
    task_type: str,
    question: str,
    evidence: list[dict],
    *,
    model_decision: str = "unknown",
    model_review: bool | None = None,
) -> ControlAssessment:
    """Generic policy-semantic governance controller.

    It uses only task type, question, retrieved policy evidence, and an optional
    model tie-breaker. It has no access to task IDs, expected answers, expected
    decisions, or expert labels.
    """
    tt = (task_type or "").strip().lower()
    q = question or ""
    reqs = extract_requirements(q, evidence)
    strong = [r for r in reqs if r.overlap >= 0.12]
    top = strong[:6]

    conflict = tt == "policy_conflict_detection" or bool(_CONFLICT_PATTERN.search(q))
    question_missing = bool(_MISSING_PATTERN.search(q))
    prohibited = [r for r in strong if r.modality == "prohibited"]
    mandatory = [r for r in strong if r.modality == "mandatory"]
    permitted = [r for r in strong if r.modality == "permitted"]
    review_rules = [r for r in strong if _REVIEW_PATTERN.search(r.text)]
    missing_mandatory = question_missing and bool(mandatory)

    if not evidence:
        return ControlAssessment(
            "unknown", True, 0.20,
            "No trusted policy evidence was retrieved; the controller abstains.",
            tuple(), False, conflict,
        )
    if conflict:
        return ControlAssessment(
            "needs_review", True, 0.90,
            "A policy conflict or precedence question requires governed resolution or escalation.",
            tuple(top), False, True,
        )
    if prohibited:
        return ControlAssessment(
            "not_allowed", bool(review_rules), 0.92,
            "Retrieved policy contains an applicable prohibition.",
            tuple(top), False, False,
        )
    if missing_mandatory:
        return ControlAssessment(
            "not_allowed", bool(review_rules), 0.88,
            "The scenario omits a control that retrieved policy expresses as mandatory.",
            tuple(top), True, False,
        )
    if review_rules:
        return ControlAssessment(
            "needs_review", True, 0.86,
            "Applicable policy evidence explicitly requires review, approval, oversight, or escalation.",
            tuple(top), False, False,
        )
    if mandatory:
        return ControlAssessment(
            "conditional", False, 0.80,
            "Applicable policy evidence imposes mandatory conditions but no explicit omission is established.",
            tuple(top), False, False,
        )
    if permitted:
        return ControlAssessment(
            "allowed", False, 0.76,
            "Applicable policy evidence explicitly permits the action and no conflicting mandatory condition was detected.",
            tuple(top), False, False,
        )

    md = (model_decision or "unknown").strip().lower()
    if md not in DECISIONS:
        md = "unknown"
    if md in {"not_allowed", "needs_review", "conditional"}:
        return ControlAssessment(
            md,
            bool(model_review) or md == "needs_review",
            0.58,
            "No decisive normative clause was extracted; a conservative model decision is retained as a tie-breaker.",
            tuple(top), False, False,
        )
    return ControlAssessment(
        "unknown", True, 0.35,
        "Evidence was retrieved but no sufficiently matched normative clause established a decision.",
        tuple(top), False, False,
    )
