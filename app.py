"""Simple FastAPI app for email classification."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import logging
import numpy as np
from src.email_classifier.models.adapter import classify_email, classify_email_with_confidence
import joblib
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define request model
class EmailRequest(BaseModel):
    text: str

# Define response model
class EmailResponse(BaseModel):
    category: str
    confidence: float = None

class DebugResponse(BaseModel):
    category: str
    confidence: float
    all_probabilities: dict
    text_length: int
    masked_text: str = None

# Create FastAPI app
app = FastAPI(
    title="Email Classification API",
    description="API for email classification",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Email Classification API is running",
        "endpoints": {
            "classify": "/classify",
            "debug": "/debug"
        }
    }

# Classification endpoint
@app.post("/classify", response_model=EmailResponse)
async def classify(request: EmailRequest):
    """
    Classify an email text.
    
    Args:
        request: The request containing email text
        
    Returns:
        Classification result
    """
    try:
        text = request.text
        logger.info(f"Received text for classification: {text[:50]}...")
        
        # Get classification with confidence
        category, confidence = classify_email_with_confidence(text)
        
        logger.info(f"Classification result: {category} with confidence {confidence:.4f}")
        
        return EmailResponse(
            category=category,
            confidence=confidence
        )
    
    except Exception as e:
        logger.error(f"Error classifying email: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error classifying email: {str(e)}")

# Debug endpoint to show all probabilities
@app.post("/debug", response_model=DebugResponse)
async def debug(request: EmailRequest):
    """
    Detailed debug information about classification.
    
    Args:
        request: The request containing email text
        
    Returns:
        Detailed classification results
    """
    try:
        from src.email_classifier.utils.pii_masker import mask_pii
        
        text = request.text
        logger.info(f"DEBUG: Received text of length {len(text)}")
        
        # Get the model and vectorizer
        model_dir = "./model"
        vectorizer = joblib.load(os.path.join(model_dir, "vectorizer.joblib"))
        model = joblib.load(os.path.join(model_dir, "classifier.joblib"))
        
        # Load label map
        with open(os.path.join(model_dir, "label_map.json"), 'r') as f:
            label_map_raw = json.load(f)
            label_map = {int(k): v for k, v in label_map_raw.items()}
        
        # Mask PII (optional)
        masked_text, entities = mask_pii(text)
        
        # Vectorize the text
        X = vectorizer.transform([text])
        
        # Get class probabilities
        probs = model.predict_proba(X)[0]
        
        # Get predicted class
        predicted_class_idx = np.argmax(probs)
        confidence_value = probs[predicted_class_idx]
        
        # Map to category name
        # Find key in label_map that has this index
        category = label_map.get(predicted_class_idx, "uncategorized")
        
        # Create probability dict
        prob_dict = {label_map.get(i, f"class_{i}"): float(p) for i, p in enumerate(probs)}
        
        return DebugResponse(
            category=category,
            confidence=confidence_value,
            all_probabilities=prob_dict,
            text_length=len(text),
            masked_text=masked_text
        )
        
    except Exception as e:
        logger.error(f"Error in debug: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in debug: {str(e)}")

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 5000))
    
    # Run the API
    uvicorn.run(app, host="0.0.0.0", port=port)