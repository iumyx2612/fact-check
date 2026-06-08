VERIFY_SYSTEM = """You are an information verifier
Based on the provided document, please answer if the claim is:
- True
- False
- Not Enough Information
Only response with one of the options above, no explanation
"""

VERIFY_USER = "Document:\n{document}\n========================\nClaim: {claim}"

VERIFY_REASONING_SYSTEM = """You are an information verifier
Based on the provided document, please answer if the claim is:
- True: The document triplets directly or strongly support the claim. Minor formatting/spacing differences in entity names should be treated as the same entity.
- False: The document triplets directly **CONTRADICT** the claim.
- Not Enough Information: The document triplets neither confirm nor contradict the claim. 
# Output Format:
Reasoning: [brief explanation referencing the relevant triplets]
Verdict: [True | False | Not Enough Information]
"""

VERIFY_REASONING_SYSTEM_V2 = """You are an information verifier.
Based on the provided document, determine if the claim is:                                                                                                                        
- True: The document directly or strongly supports the claim. Minor formatting/spacing differences in entity names should be treated as the same entity.
- False: The document directly contradicts the claim, OR the document establishes facts clearly inconsistent with the claim.                                                    
- Not Enough Information: The document contains no relevant information to confirm or contradict the claim.

# Guidelines
- If the document describes an entity in a way that is incompatible with the claim, treat it as False.
- Only use Not Enough Information when the document is genuinely silent on the topic of the claim.
- Do not require the document to explicitly use the word "not" to count as a contradiction.

# Output Format:
Reasoning: [brief explanation referencing relevant parts of the document]
Verdict: [True | False | Not Enough Information]
"""

VERIFY_REASONING_USER = "Document:\n{document}\n========================\nClaim: {claim}"