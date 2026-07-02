from policy_as_skill.data_loader import PolicyDocument
from policy_as_skill.retrieval import PolicyRetriever

def test_retrieval_finds_data_protection():
    docs=[PolicyDocument('p.md','## data-protection\nPersonal data must be minimized.\n## external-cloud\nCloud needs approval.')]
    r=PolicyRetriever(docs)
    chunks=r.retrieve('personal data minimization', top_k=1)
    assert chunks[0].chunk_id == 'data-protection'
    assert chunks[0].score > 0
