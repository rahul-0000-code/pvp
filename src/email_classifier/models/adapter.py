"""Model adapter to support multiple model types."""
import os
import logging
import json
import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelAdapter:
    """Adapter that can work with different model types."""
    
    def __init__(self, model_dir: str = "./model", confidence_threshold: float = 0.3):
        """
        Initialize the model adapter.
        
        Args:
            model_dir: Directory where model files are stored
            confidence_threshold: Threshold for classification confidence
        """
        self.model_dir = model_dir
        self.model_type = None  # 'bert' or 'sklearn'
        self.model = None
        self.tokenizer = None
        self.vectorizer = None
        self.label_map = None
        self.confidence_threshold = confidence_threshold
        
    def load_model(self) -> bool:
        """
        Load the appropriate model based on what's available.
        
        Returns:
            True if a model was loaded successfully, False otherwise
        """
        try:
            # First check if we have a scikit-learn model
            if os.path.exists(os.path.join(self.model_dir, "classifier.joblib")) and \
               os.path.exists(os.path.join(self.model_dir, "vectorizer.joblib")):
                return self._load_sklearn_model()
                
            # Then try to load BERT model
            elif os.path.exists(os.path.join(self.model_dir, "pytorch_model.bin")) or \
                 os.path.exists(os.path.join(self.model_dir, "model.safetensors")):
                return self._load_bert_model()
                
            else:
                logger.error(f"No model files found in {self.model_dir}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
            
    def _load_bert_model(self) -> bool:
        """Load BERT-based models."""
        try:
            from transformers import BertTokenizer, BertForSequenceClassification
            
            logger.info(f"Loading BERT model from {self.model_dir}")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)
            self.model = BertForSequenceClassification.from_pretrained(self.model_dir)
            self.model.eval()  # Set to evaluation mode
            
            # Load label map
            self._load_label_map()
            
            self.model_type = 'bert'
            logger.info("BERT model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading BERT model: {e}")
            return False
            
    def _load_sklearn_model(self) -> bool:
        """Load scikit-learn-based models."""
        try:
            import joblib
            
            logger.info(f"Loading scikit-learn model from {self.model_dir}")
            self.vectorizer = joblib.load(os.path.join(self.model_dir, "vectorizer.joblib"))
            self.model = joblib.load(os.path.join(self.model_dir, "classifier.joblib"))
            
            # Try to load label map from joblib first
            try:
                sklearn_label_map_path = os.path.join(self.model_dir, "sklearn_label_map.joblib")
                if os.path.exists(sklearn_label_map_path):
                    self.label_map = joblib.load(sklearn_label_map_path)
                else:
                    self._load_label_map()
            except:
                self._load_label_map()
            
            self.model_type = 'sklearn'
            logger.info("scikit-learn model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading scikit-learn model: {e}")
            return False
    
    def _load_label_map(self) -> None:
        """Load the label mapping from JSON file."""
        try:
            label_map_path = os.path.join(self.model_dir, "label_map.json")
            if os.path.exists(label_map_path):
                with open(label_map_path, 'r') as f:
                    # The file has format: {"0": "Request", "1": "Change", "2": "Problem", "3": "Incident"}
                    # We want: {"Request": 0, "Change": 1, "Problem": 2, "Incident": 3}
                    raw_label_map = json.load(f)
                    # Invert the map for easier lookup
                    self.label_map = {v: int(k) for k, v in raw_label_map.items()}
                    logger.info(f"Loaded and inverted label map: {self.label_map}")
            else:
                logger.warning(f"Label map not found at {label_map_path}")
                self.label_map = None
        except Exception as e:
            logger.warning(f"Error loading label map: {e}")
            self.label_map = None
    
    def predict(self, text: str) -> str:
        """
        Classify text using the loaded model.
        
        Args:
            text: The text to classify
            
        Returns:
            Predicted category as string
        """
        # Load model if not already loaded
        if self.model is None:
            if not self.load_model():
                return "uncategorized"
        
        category, confidence = self.predict_with_confidence(text)
        return category
    
    def predict_with_confidence(self, text: str) -> Tuple[str, float]:
        """
        Classify text and return category with confidence.
        
        Args:
            text: The text to classify
            
        Returns:
            Tuple of (category, confidence)
        """
        try:
            # Load model if not already loaded
            if self.model is None:
                if not self.load_model():
                    return "uncategorized", 0.0
            
            # Use appropriate prediction method based on model type
            if self.model_type == 'bert':
                return self._predict_with_bert(text)
            elif self.model_type == 'sklearn':
                return self._predict_with_sklearn(text)
            else:
                logger.error("Unknown model type or model not loaded")
                return "uncategorized", 0.0
                
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return "uncategorized", 0.0
    
    def _predict_with_bert(self, text: str) -> Tuple[str, float]:
        """Make prediction using BERT model."""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get confidence and class
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence, predicted_class = torch.max(probs, dim=1)
            
            confidence_value = confidence.item()
            predicted_class_idx = predicted_class.item()
            
            logger.info(f"BERT prediction: class={predicted_class_idx}, confidence={confidence_value:.4f}")
            
            # Check confidence threshold
            if confidence_value < self.confidence_threshold:
                return "uncategorized", confidence_value
            
            # Map to category name
            if self.label_map and predicted_class_idx in self.label_map:
                category = self.label_map[predicted_class_idx]
            else:
                # Fallback to position-based mapping
                categories = ["Problem", "Incident", "Change", "Request"]
                category = categories[predicted_class_idx] if predicted_class_idx < len(categories) else "uncategorized"
            
            return category, confidence_value
            
        except Exception as e:
            logger.error(f"Error in BERT prediction: {e}")
            return "uncategorized", 0.0
    
    def _predict_with_sklearn(self, text: str) -> Tuple[str, float]:
        """Make prediction using scikit-learn model."""
        try:
            # Vectorize the text
            X = self.vectorizer.transform([text])
            
            # Get class probabilities
            probs = self.model.predict_proba(X)[0]
            predicted_class_idx = np.argmax(probs)
            confidence_value = probs[predicted_class_idx]
            
            logger.info(f"sklearn prediction: class={predicted_class_idx}, confidence={confidence_value:.4f}")
            
            # Check confidence threshold
            if confidence_value < self.confidence_threshold:
                return "uncategorized", confidence_value
            
            # Map numerical index to actual category name
            # Load label map if not loaded
            if self.label_map is None:
                self._load_label_map()
            
            # We need to find the key in the label map that has the value equal to our predicted_class_idx
            category = "uncategorized"
            if self.label_map:
                # Our label map is inverted (category name -> index)
                # Find the category name that matches our predicted index
                for cat_name, idx in self.label_map.items():
                    if idx == predicted_class_idx:
                        category = cat_name
                        logger.info(f"Found category {category} for class {predicted_class_idx}")
                        break
            
            if category == "uncategorized":
                logger.warning(f"Could not find category for class {predicted_class_idx} in label map {self.label_map}")
                # Try to get from model classes
                try:
                    category = self.model.classes_[predicted_class_idx]
                except:
                    # Fallback to position-based mapping
                    categories = ["Problem", "Incident", "Change", "Request"]
                    category = categories[predicted_class_idx] if predicted_class_idx < len(categories) else "uncategorized"
            
            return category, confidence_value
            
        except Exception as e:
            logger.error(f"Error in sklearn prediction: {e}")
            return "uncategorized", 0.0

# Create a singleton instance
adapter = ModelAdapter()

def classify_email(text: str) -> str:
    """
    Classify an email into a category.
    
    This is a convenience function that uses the ModelAdapter singleton.
    
    Args:
        text: The email text to classify.
        
    Returns:
        The predicted category as a string.
    """
    return adapter.predict(text)

def classify_email_with_confidence(text: str) -> Tuple[str, float]:
    """
    Classify an email and return both category and confidence score.
    
    Args:
        text: The email text to classify.
        
    Returns:
        Tuple containing the predicted category and confidence score.
    """
    return adapter.predict_with_confidence(text) 