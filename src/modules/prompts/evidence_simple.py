EVIDENCE_SIMPLE_USER = """The **bold** parts in the evidence below are the actual supporting elements for the claim.

Evidence:
{evidence}

Choose your answer: based on the evidence above can we conclude that "{claim}"?
OPTIONS:
- Yes: The evidence SUPPORTS the claim
- No: The evidence CONTRADICTS the claim
- Not Enough Information: The evidence does not have enough information to support the claim
I think the answer is [one of Yes, No, Not Enough Information without explanation]"""

EVIDENCE_SIMPLE_REASONING_USER = """The **bold** parts in the evidence below are the actual supporting elements for the claim.

Evidence:
{evidence}

Choose your answer: based on the evidence above can we conclude that "{claim}"?
OPTIONS:
- Yes: The evidence SUPPORTS the claim
- No: The evidence CONTRADICTS the claim
- Not Enough Information: The evidence does not have enough information to support the claim
Please answer using the following template
```
Reasoning: [Reason for the answer]
Answer: [one of Yes, No, Not Enough Information]
```
"""