from dataclasses import dataclass
import math, re
from collections import Counter
from .data_loader import PolicyDocument
from .utils import tokens

@dataclass
class PolicyChunk:
    source: str
    chunk_id: str
    text: str
    score: float = 0.0

class PolicyRetriever:
    def __init__(self, docs: list[PolicyDocument]):
        self.chunks: list[PolicyChunk] = []
        for d in docs:
            sections = re.split(r'(?=^##\s+)', d.text, flags=re.M)
            for i, sec in enumerate(s.strip() for s in sections if s.strip()):
                m = re.search(r'^##\s+([\w-]+)', sec, re.M)
                cid = m.group(1) if m else f'chunk-{i}'
                self.chunks.append(PolicyChunk(d.source, cid, sec))
        self.doc_tokens=[tokens(c.text) for c in self.chunks]
        self.idf={}
        vocab=set().union(*self.doc_tokens) if self.doc_tokens else set()
        for term in vocab:
            df=sum(1 for dt in self.doc_tokens if term in dt)
            self.idf[term]=math.log((1+len(self.chunks))/(1+df))+1

    def _score(self, query: str, c: PolicyChunk) -> float:
        qt=Counter(tokens(query)); dt=Counter(tokens(c.text)); score=0.0
        for term, qtf in qt.items():
            score += qtf * dt.get(term,0) * self.idf.get(term,1.0)
        norm=math.sqrt(sum(v*v for v in qt.values()))*math.sqrt(sum(v*v for v in dt.values()))
        return score/max(norm,1e-9)

    def retrieve(self, query: str, top_k: int = 4) -> list[PolicyChunk]:
        scored=[(self._score(query,c),c) for c in self.chunks]
        return [PolicyChunk(c.source,c.chunk_id,c.text,float(s)) for s,c in sorted(scored,key=lambda x:x[0], reverse=True)[:top_k]]

    def keyword(self, query: str, top_k: int = 4) -> list[PolicyChunk]:
        qt=tokens(query); scored=[(len(qt & tokens(c.text)), c) for c in self.chunks]
        return [PolicyChunk(c.source,c.chunk_id,c.text,float(s)) for s,c in sorted(scored,key=lambda x:x[0], reverse=True)[:top_k]]
