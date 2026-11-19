"""Strict response contracts shared by the graph and HTTP API."""
from pydantic import BaseModel, ConfigDict, Field, model_validator


class GuardrailDecision(BaseModel):
    # strict rejects coercion (e.g. "false" to False); extra rejects unexpected keys.
    """Exact contract required before any research is authorized."""
    model_config = ConfigDict(strict=True, extra="forbid")
    allowed: bool
    reason: str

    @model_validator(mode="after")
    def require_block_reason(self):
        if not self.allowed and not self.reason.strip():
            raise ValueError("Blocked requests require a reason")
        return self


class HumanReview(BaseModel):
    """Shared contract for the HTTP request and resumed graph decision."""
    model_config = ConfigDict(strict=True, extra="forbid")
    approved: bool
    feedback: str = ""

    @model_validator(mode="after")
    def require_revision_feedback(self):
        self.feedback = self.feedback.strip()
        if not self.approved and not self.feedback:
            raise ValueError("Revision feedback is required")
        return self


class RevisedRequest(BaseModel):
    """A complete replacement request; empty or non-string rewrites are invalid."""
    model_config = ConfigDict(strict=True, extra="forbid", str_strip_whitespace=True)
    updated_request: str = Field(min_length=1)
