VERIFY_TRIPLE_USER = """
Given a claim and supporting evidence, determine whether the evidence supports, refutes, or provides not enough information to verify the claim.

Claim: {claim}

Evidence:
{evidence}

Based only on the evidence provided above, determine if the evidence:
- SUPPORTS the claim (the evidence confirms the claim is true)
- REFUTES the claim (the evidence contradicts the claim or shows it is false)
- NOT ENOUGH INFORMATION (the evidence neither confirms nor denies the claim)

Provide your answer as a single word: SUPPORT, REFUTE, or NOT ENOUGH INFORMATION
"""

VERIFY_TRIPLE_WITH_CONTEXT_USER = """
Given a claim and supporting evidence, determine whether the evidence supports, refutes, or provides not enough information to verify the claim.

Claim: {claim}

Gold Evidence (for reference):
{gold_evidence}

Retrieved Evidence:
{retrieved_evidence}

Based on the retrieved evidence, determine if it:
- SUPPORTS the claim (the evidence confirms the claim is true)
- REFUTES the claim (the evidence contradicts the claim or shows it is false)
- NOT ENOUGH INFORMATION (the evidence neither confirms nor denies the claim)

Provide your answer as a single word: SUPPORT, REFUTE, or NOT ENOUGH INFORMATION
"""

# Binary verification — strings aligned with ``tests/original_graphcheck.OpenAIBaseModel.verify``
GRAPH_CHECK_VERIFY_NO_EVIDENCE = (
    "Is this claim true or false? Claim: {claim}\nAnswer only 'true' or 'false':"
)

GRAPH_CHECK_VERIFY_WITH_EVIDENCE = """Based on this evidence:
{evidence}

Is this claim true or false? Claim: {claim}
Answer only 'true' or 'false':"""

# Optional gold context (not in the original tracer; same yes/no question over retrieved text)
GRAPH_CHECK_VERIFY_WITH_GOLD_AND_RETRIEVED = """Reference (gold, optional):
{gold_evidence}

Based on this evidence:
{retrieved_evidence}

Is this claim true or false? Claim: {claim}
Answer only 'true' or 'false':"""

# Backwards-compatible names for imports
BINARY_VERIFY_TRIPLE_USER = GRAPH_CHECK_VERIFY_WITH_EVIDENCE
BINARY_VERIFY_TRIPLE_WITH_CONTEXT_USER = GRAPH_CHECK_VERIFY_WITH_GOLD_AND_RETRIEVED
