VERIFY_SYSTEM = """You are an information verifier
Based on the provided document, please answer if the claim is:
- True
- False
- Not Enough Information
Only response with one of the options above, no explanation
"""

VERIFY_USER = "Document:\n{document}\n========================\nClaim: {claim}"