ARCHITECTURE = """flowchart LR
A[Policy Repository]-->B[Trusted Retrieval]
B-->C[Skill Registry]
C-->D[Agent Planner]
D-->E[Ollama Reasoning]
E-->F[Human Review]
F-->G[Audit Trail]
G-->H[Evaluation]
H-->I[HTML Report]
"""
LOOP = """flowchart TD
A[Policy Knowledge]-->B[Trusted Retrieval]
B-->C[Reasoning]
C-->D[Tool / Action Support]
D-->E[Human Review]
E-->F[Audit Trail]
F-->G[Evaluation]
G-->H[Policy Improvement]
H-->A
"""
