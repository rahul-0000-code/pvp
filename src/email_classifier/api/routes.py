"""API routes for the email classification service."""
from fastapi import APIRouter, HTTPException, Query
import logging
from ..utils.pii_masker import mask_pii
from ..models.classifier import classify_email, classify_email_with_confidence
from .api_models import EmailRequest, EmailResponse, MaskedEntity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["classification"])


@router.post("/classify", response_model=EmailResponse)
async def process_email(
    request: EmailRequest,
    include_confidence: bool = Query(
        False, 
        description="Whether to include confidence score in debug info"
    )
):
    """
    Process an email: mask PII and classify it.
    
    Args:
        request: The email request containing the email body.
        include_confidence: Whether to include confidence score in debug info.
        
    Returns:
        EmailResponse with masked entities, masked email, and classification.
        
    Raises:
        HTTPException: If there's an error processing the email.
    """
    try:
        # 1. Get the input email
        email_text = request.email_body
        
        logger.info(f"Processing email of length {len(email_text)} characters")
        
        # 2. Mask PII and get entity information
        masked_email, raw_entities = mask_pii(email_text)
        
        logger.info(f"Masked email content: {masked_email[:100]}...")
        logger.info(f"Found {len(raw_entities)} PII entities")
        
        # 3. Classify the masked email (with or without confidence)
        if include_confidence:
            category, confidence = classify_email_with_confidence(masked_email)
            logger.info(f"Classification result: {category} with confidence {confidence:.4f}")
        else:
            category = classify_email(masked_email)
            logger.info(f"Classification result: {category}")
        
        # 4. Convert entities to the required format
        formatted_entities = [
            MaskedEntity(
                position=[entity['start'], entity['end']],
                classification=entity['type'],
                entity=entity['value']
            ) 
            for entity in raw_entities
        ]
        
        # 5. Prepare the response
        response = EmailResponse(
            input_email_body=email_text,
            list_of_masked_entities=formatted_entities,
            masked_email=masked_email,
            category_of_the_email=category
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing email: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing email: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Status message.
    """
    return {"status": "healthy"} 