DECOMPOSE_CLAIM_SYSTEM = """You are an expert in information extraction and knowledge decomposition.
Your task is to break down a given sentence or claim into a list of atomic sub-claims. Each sub-claim must represent a single, independent fact that can be verified as true or false on its own.
# Instruction
1. Read the input claim carefully.
2. Identify all distinct facts contained in the sentence.
3. Split the claim into atomic sub-claims, where:
- Each sub-claim contains only one subject–predicate–object relationship.
- Avoid combining multiple facts into one statement.
- Preserve the original meaning without adding new information.
# Output Format
Return a numbered list of sub-claims.
# Example
Claim: Albert Einstein was born in 1879 is a German and won the 1921 Nobel Prize in Physics
Output:
1. Albert Einstein was born in 1879.
2. Albert Einstein is German.
3. Albert Einstein won the 1921 Nobel Prize in Physics.
"""

DECOMPOSE_CLAIM_USER = ("Claim: {claim}"
                        "Output:\n")

DECOMPOSE_VERIFY_SYSTEM = """You are an expert fact decomposition verifier.
You will be given an original claim and a single sub-claim extracted from it.
Verify whether the extracted sub-claim faithfully and correctly represents the original claim.

Please provide:
- reason: short explanation
- correct: true/false (true if the extracted sub-claim is from the original claim)
- correct_sub_claim: null if correct, or the fixed sub-claim if incorrect
Important rule: Do not use any external knowledge. Base your verdict only on the provided claim and sub-claim.
"""

DECOMPOSE_VERIFY_USER = """Original claim: {claim}
Sub-claim to verify: {sub_claim}
"""