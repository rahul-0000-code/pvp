"""Email classification module."""
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default category map
DEFAULT_CATEGORY_MAP = {
    0: "request",
    1: "incident", 
    2: "problem",
    3: "billing_issue",
    4: "technical_support",
    5: "account_management",
    6: "general_inquiry"
}

# Confidence threshold for classification
# If the model's confidence is below this, return "uncategorized"
CONFIDENCE_THRESHOLD = 0.6


class EmailClassifier:
    """Class for classifying emails into categories."""
    
    def __init__(self, model_dir: str = "./model", confidence_threshold: float = CONFIDENCE_THRESHOLD):
        """
        Initialize the email classifier.
        
        Args:
            model_dir: Directory where the model and tokenizer are stored.
            confidence_threshold: Threshold for classification confidence (0.0 to 1.0)
        """
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.label_map = None
        self.confidence_threshold = confidence_threshold
        
    def load_model(self) -> bool:
        """
        Load the pretrained model and tokenizer.
        
        Returns:
            True if the model was loaded successfully, False otherwise.
        """
        try:
            if not os.path.exists(self.model_dir):
                logger.warning(f"Model directory {self.model_dir} not found.")
                return False
                
            logger.info(f"Loading model from {self.model_dir}")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)
            self.model = BertForSequenceClassification.from_pretrained(self.model_dir)
            
            # Load label map if exists
            label_map_path = os.path.join(self.model_dir, "label_map.json")
            if os.path.exists(label_map_path):
                try:
                    df = pd.read_json(label_map_path)
                    self.label_map = {v: k for k, v in df[0].items()} if 0 in df else None
                    logger.info(f"Loaded label map: {self.label_map}")
                except Exception as e:
                    logger.warning(f"Could not load label map, using default: {e}")
                    self.label_map = None
            else:
                logger.warning(f"Label map file not found at {label_map_path}, using default categories")
            
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, text: str) -> str:
        """
        Classify the given text into a category.
        
        Args:
            text: The text to classify.
            
        Returns:
            The predicted category as a string.
        """
        try:
            # Load model if not already loaded
            if self.model is None or self.tokenizer is None:
                logger.info("Model not loaded, attempting to load")
                if not self.load_model():
                    logger.warning("Failed to load model, returning uncategorized")
                    return "uncategorized"
            
            # Tokenize input
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get predicted class and confidence
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence, predicted_class = torch.max(probs, dim=1)
            
            # Convert to Python types
            confidence_value = confidence.item()
            predicted_class_idx = predicted_class.item()
            
            logger.info(f"Predicted class: {predicted_class_idx} with confidence: {confidence_value:.4f}")
            
            # Check confidence threshold
            if confidence_value < self.confidence_threshold:
                logger.info(f"Confidence {confidence_value:.4f} below threshold {self.confidence_threshold}, returning uncategorized")
                return "uncategorized"
            
            # Map class to category name
            if self.label_map:
                category = self.label_map.get(predicted_class_idx, "general_inquiry")
            else:
                category = DEFAULT_CATEGORY_MAP.get(predicted_class_idx, "general_inquiry")
                
            logger.info(f"Returning category: {category}")
            return category
        
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return "uncategorized"
    
    def predict_with_confidence(self, text: str) -> Tuple[str, float]:
        """
        Classify the given text and return both category and confidence.
        
        Args:
            text: The text to classify.
            
        Returns:
            Tuple containing the predicted category and confidence score.
        """
        try:
            # Load model if not already loaded
            if self.model is None or self.tokenizer is None:
                if not self.load_model():
                    return "uncategorized", 0.0
            
            # Tokenize input
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get predicted class and confidence
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence, predicted_class = torch.max(probs, dim=1)
            
            # Convert to Python types
            confidence_value = confidence.item()
            predicted_class_idx = predicted_class.item()
            
            # Check confidence threshold
            if confidence_value < self.confidence_threshold:
                return "uncategorized", confidence_value
            
            # Map class to category name
            if self.label_map:
                category = self.label_map.get(predicted_class_idx, "general_inquiry")
            else:
                category = DEFAULT_CATEGORY_MAP.get(predicted_class_idx, "general_inquiry")
                
            return category, confidence_value
        
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return "uncategorized", 0.0
    
    def train(self, data_path: str, epochs: int = 3) -> bool:
        """
        Train the model on the given dataset.
        
        Args:
            data_path: Path to the dataset CSV file.
            epochs: Number of training epochs.
            
        Returns:
            True if training was successful, False otherwise.
        """
        try:
            logger.info(f"Loading dataset from {data_path}...")
            df = pd.read_csv(data_path)
            
            # Print dataset info
            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Dataset columns: {df.columns.tolist()}")
            
            # Check if required columns exist
            if 'masked_email' not in df.columns or 'category' not in df.columns:
                available_columns = df.columns.tolist()
                logger.error(f"Required columns 'masked_email' or 'category' not found in dataset. Available columns: {available_columns}")
                # Try to adapt to available columns
                text_column = next((col for col in available_columns if 'text' in col.lower() or 'email' in col.lower() or 'content' in col.lower()), None)
                label_column = next((col for col in available_columns if 'category' in col.lower() or 'label' in col.lower() or 'class' in col.lower()), None)
                
                if text_column and label_column:
                    logger.info(f"Using alternative columns: text='{text_column}', label='{label_column}'")
                    X = df[text_column].tolist()
                    y = df[label_column].tolist()
                else:
                    logger.error("Cannot find suitable text and label columns. Please check your dataset.")
                    return False
            else:
                # Use the standard column names
                X = df['masked_email'].tolist()  # Use masked emails for training
                y = df['category'].tolist()      # Email categories
            
            # Print class distribution
            class_counts = {}
            for label in y:
                class_counts[label] = class_counts.get(label, 0) + 1
            logger.info(f"Class distribution: {class_counts}")
            
            # Convert string labels to numerical
            label_map = {cat: i for i, cat in enumerate(set(y))}
            y_encoded = [label_map[label] for label in y]
            
            logger.info(f"Label mapping: {label_map}")
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42
            )
            
            # Initialize tokenizer and model
            logger.info("Initializing BERT model...")
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=len(label_map)
            )
            
            # Tokenize data
            logger.info("Tokenizing data...")
            train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
            val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=128)
            
            # Create PyTorch datasets
            class EmailDataset(torch.utils.data.Dataset):
                def __init__(self, encodings, labels):
                    self.encodings = encodings
                    self.labels = labels
                    
                def __getitem__(self, idx):
                    item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                    item['labels'] = torch.tensor(self.labels[idx])
                    return item
                
                def __len__(self):
                    return len(self.labels)
            
            train_dataset = EmailDataset(train_encodings, y_train)
            val_dataset = EmailDataset(val_encodings, y_val)
            
            # Training settings
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {device}")
            model.to(device)
            
            training_args = {
                'batch_size': 16,
                'epochs': epochs,
                'learning_rate': 5e-5
            }
            
            # Create data loaders
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=training_args['batch_size'], shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=training_args['batch_size'])
            
            # Initialize optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=training_args['learning_rate'])
            
            # Training loop
            logger.info("Starting training...")
            for epoch in range(training_args['epochs']):
                model.train()
                train_loss = 0
                
                for batch in train_loader:
                    optimizer.zero_grad()
                    
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
                        
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                        
                        val_loss += loss.item()
                        
                        # Calculate accuracy
                        predictions = torch.argmax(outputs.logits, dim=1)
                        correct += (predictions == labels).sum().item()
                        total += labels.size(0)
                
                # Log metrics
                logger.info(f"Epoch {epoch+1}/{training_args['epochs']}")
                logger.info(f"Train Loss: {train_loss/len(train_loader):.4f}")
                logger.info(f"Val Loss: {val_loss/len(val_loader):.4f}")
                logger.info(f"Val Accuracy: {correct/total:.4f}")
            
            # Create model directory if it doesn't exist
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
                
            # Save model and tokenizer
            logger.info(f"Saving model to {self.model_dir}...")
            model.save_pretrained(self.model_dir)
            tokenizer.save_pretrained(self.model_dir)
            
            # Save label mapping
            logger.info(f"Saving label mapping to {self.model_dir}/label_map.json")
            pd.DataFrame([label_map]).to_json(os.path.join(self.model_dir, "label_map.json"))
            
            # Update instance variables
            self.model = model
            self.tokenizer = tokenizer
            self.label_map = {v: k for k, v in label_map.items()}
            
            logger.info("Training completed successfully!")
            return True
        
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return False


# Create a singleton instance
classifier = EmailClassifier()

def classify_email(text: str) -> str:
    """
    Classify an email into a category.
    
    This is a convenience function that uses the EmailClassifier singleton.
    
    Args:
        text: The email text to classify.
        
    Returns:
        The predicted category as a string.
    """
    return classifier.predict(text)

def classify_email_with_confidence(text: str) -> Tuple[str, float]:
    """
    Classify an email and return both category and confidence score.
    
    Args:
        text: The email text to classify.
        
    Returns:
        Tuple containing the predicted category and confidence score.
    """
    return classifier.predict_with_confidence(text)

def train_model(data_path: str = "data/emails.csv", epochs: int = 3) -> bool:
    """
    Train the classification model.
    
    This is a convenience function that uses the EmailClassifier singleton.
    
    Args:
        data_path: Path to the dataset CSV file.
        epochs: Number of training epochs.
        
    Returns:
        True if training was successful, False otherwise.
    """
    return classifier.train(data_path, epochs) 