"""Pydantic models for the API."""
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Any


class EmailRequest(BaseModel):
    """Request model for email classification."""
    email_body: str = Field(
        ...,
        description="The email body text to be classified and masked for PII"
    )


class MaskedEntity(BaseModel):
    """Model for a masked PII entity."""
    position: Tuple[int, int] = Field(
        ...,
        description="Start and end positions of the entity in the original text"
    )
    classification: str = Field(
        ...,
        description="Type of the entity (full_name, email, phone_number, etc.)"
    )
    entity: str = Field(
        ...,
        description="The original value of the entity"
    )


class EmailResponse(BaseModel):
    """Response model for email classification."""
    input_email_body: str = Field(
        ...,
        description="The body text"
    )
    list_of_masked_entities: List[MaskedEntity] = Field(
        ...,
        description="List of detected PII entities with their positions and types"
    )
    masked_email: str = Field(
        ...,
        description="The email with PII entities masked"
    )
    category_of_the_email: str = Field(
        ...,
        description="The predicted category of the email"
    ) 