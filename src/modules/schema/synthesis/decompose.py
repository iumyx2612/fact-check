from pydantic import BaseModel, Field


class SubClaimVerdict(BaseModel):
    reason: str = Field(description="Short explanation for the verdict")
    correct: bool = Field(description="Whether the sub-claim correctly extracted from the original claim")
    correct_sub_claim: str | None = Field(
        description="The corrected sub-claim only if incorrect, otherwise null"
    )