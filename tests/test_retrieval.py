from policy_as_skill.data_loader import PolicyDocument
from policy_as_skill.retrieval import PolicyRetriever


def test_retrieval_finds_data_protection():
    docs = [PolicyDocument('p.md', '## data-protection\nPersonal data must be minimized.\n## external-cloud\nCloud needs approval.')]
    r = PolicyRetriever(docs)
    chunks = r.retrieve('personal data minimization', top_k=1)
    assert chunks[0].chunk_id == 'data-protection'
    assert chunks[0].score > 0
    assert 'data-protection' in chunks[0].tags


def test_hybrid_scope_bonus_finds_human_review():
    docs = [PolicyDocument('p.md', '## human-review\nSensitive cases require human oversight.\n## pilot\nLow risk pilots may use monitoring.')]
    r = PolicyRetriever(docs)
    chunks = r.retrieve('external citizen image', top_k=1, scope_terms=['human-review'], mode='hybrid')
    assert chunks[0].chunk_id == 'human-review'
