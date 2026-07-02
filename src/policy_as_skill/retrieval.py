from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .data_loader import PolicyDocument
from .utils import token_list, tokens


TAG_SYNONYMS = {
    "data-protection": {"data", "privacy", "personal", "citizen", "image", "sensitive", "anonymized", "anonymised"},
    "external-cloud": {"external", "cloud", "vendor", "service", "upload", "outside"},
    "human-review": {"human", "review", "oversight", "approval", "escalate"},
    "high-risk-ai": {"high", "risk", "benefit", "eligibility", "essential", "services", "ranking"},
    "transparency": {"transparent", "transparency", "notice", "explain", "disclose"},
    "audit-trail": {"audit", "trace", "log", "timestamp", "record", "reproduce", "version"},
    "accountability": {"owner", "responsible", "accountability", "accountable", "official"},
    "policy-conflict": {"conflict", "stricter", "precedence", "less", "strict"},
    "pilot-deployment": {"pilot", "deployment", "rollback", "monitoring"},
    "model-governance": {"model", "governance", "configuration", "prompt", "skill"},
    "evidence-grounding": {"evidence", "citation", "cited", "grounded", "unsupported"},
    "access-control": {"access", "role", "permission", "restricted", "scope"},
    "pilot-policy-v2": {"v2", "current", "owner", "rollback", "pilot"},
}


@dataclass
class PolicyChunk:
    source: str
    chunk_id: str
    text: str
    score: float = 0.0
    version: str = "v1"
    sha256: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def citation_id(self) -> str:
        return f"{self.source}#{self.chunk_id}@{self.version}"

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "chunk_id": self.chunk_id,
            "citation_id": self.citation_id,
            "version": self.version,
            "sha256": self.sha256,
            "tags": self.tags,
            "score": round(self.score, 6),
            "text": self.text,
        }


class PolicyRetriever:
    def __init__(self, docs: list[PolicyDocument]):
        self.chunks: list[PolicyChunk] = []
        for d in docs:
            sections = re.split(r"(?=^##\s+)", d.text, flags=re.M)
            for i, sec in enumerate(s.strip() for s in sections if s.strip()):
                m = re.search(r"^##\s+([^\n]+)", sec, re.M)
                raw_id = m.group(1).strip() if m else f"chunk-{i}"
                cid = re.sub(r"[^a-zA-Z0-9_.-]+", "-", raw_id.lower()).strip("-") or f"chunk-{i}"
                self.chunks.append(
                    PolicyChunk(
                        source=d.source,
                        chunk_id=cid,
                        text=sec,
                        version=d.version,
                        sha256=d.sha256[:12],
                        tags=self._infer_tags(cid, sec),
                    )
                )
        self.doc_token_lists = [token_list(c.text) for c in self.chunks]
        self.doc_tokens = [set(t) for t in self.doc_token_lists]
        self.avg_len = sum(len(t) for t in self.doc_token_lists) / max(1, len(self.doc_token_lists))
        self.idf = {}
        vocab = set().union(*self.doc_tokens) if self.doc_tokens else set()
        for term in vocab:
            df = sum(1 for dt in self.doc_tokens if term in dt)
            self.idf[term] = math.log(1 + (len(self.chunks) - df + 0.5) / (df + 0.5))

    def _infer_tags(self, chunk_id: str, text: str) -> list[str]:
        low = f"{chunk_id} {text}".lower()
        out = set()
        for tag, syns in TAG_SYNONYMS.items():
            if tag in low or any(s in low for s in syns):
                out.add(tag)
        return sorted(out)

    def _keyword_score(self, query: str, c: PolicyChunk) -> float:
        qt = tokens(query)
        ct = tokens(c.text) | set(c.tags)
        return float(len(qt & ct))

    def _bm25_score(self, query: str, c: PolicyChunk, idx: int) -> float:
        q = token_list(query)
        doc = self.doc_token_lists[idx]
        tf = Counter(doc)
        k1, b = 1.5, 0.75
        score = 0.0
        doc_len = len(doc)
        for term in q:
            if term not in tf:
                continue
            idf = self.idf.get(term, 0.0)
            freq = tf[term]
            denom = freq + k1 * (1 - b + b * doc_len / max(1.0, self.avg_len))
            score += idf * (freq * (k1 + 1)) / max(denom, 1e-9)
        return float(score)

    def _scope_bonus(self, c: PolicyChunk, scope_terms: list[str] | None) -> float:
        if not scope_terms:
            return 0.0
        scope = {s.lower() for s in scope_terms}
        return 0.65 * len(scope & set(c.tags)) + 0.15 * len(scope & tokens(c.text))

    def _version_bonus(self, query: str, c: PolicyChunk) -> float:
        q = query.lower()
        if "current" in q and c.version.lower() == "v2":
            return 0.7
        if c.version.lower() in q:
            return 0.5
        return 0.0

    def retrieve(self, query: str, top_k: int = 5, scope_terms: list[str] | None = None, mode: str = "bm25") -> list[PolicyChunk]:
        scored = []
        for i, c in enumerate(self.chunks):
            if mode == "keyword":
                s = self._keyword_score(query, c)
            elif mode == "hybrid":
                s = self._bm25_score(query, c, i) + 0.25 * self._keyword_score(query, c)
            else:
                s = self._bm25_score(query, c, i)
            s += self._scope_bonus(c, scope_terms) + self._version_bonus(query, c)
            scored.append((s, c))
        selected = []
        for s, c in sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]:
            selected.append(
                PolicyChunk(c.source, c.chunk_id, c.text, float(s), c.version, c.sha256, list(c.tags))
            )
        return selected

    def keyword(self, query: str, top_k: int = 5) -> list[PolicyChunk]:
        return self.retrieve(query, top_k=top_k, mode="keyword")

    def all_policy_text(self, max_chars: int = 14000) -> str:
        buf = []
        total = 0
        for c in self.chunks:
            block = f"[{c.citation_id}] tags={','.join(c.tags)}\n{c.text}\n"
            if total + len(block) > max_chars:
                break
            total += len(block)
            buf.append(block)
        return "\n".join(buf)
