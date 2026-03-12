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
