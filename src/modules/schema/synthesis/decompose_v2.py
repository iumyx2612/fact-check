from pydantic import BaseModel, Field


class SubClaimVerdict(BaseModel):
    index: int = Field(description="The sub-claim index (matches the numbered list)")
    correct: bool = Field(description="Whether the sub-claim correctly represents the original claim")
    reason: str = Field(description="Short explanation for the verdict")
    correct_sub_claim: str | None = Field(
        description="The corrected triplet (Subject -> Relation -> Object) only if incorrect, otherwise null"
    )


class DecompositionVerdict(BaseModel):
    verdicts: list[SubClaimVerdict] = Field(
        description="One verdict per sub-claim, in the same order as the input list"
    )