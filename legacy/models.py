models.py
# models.py
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Map of numerical labels to string categories
CATEGORY_MAP = {
    0: "request",
    1: "incident", 
    2: "problem",
    3: "billing_issue",
    4: "technical_support",
    5: "account_management",
    6: "general_inquiry"
}

# Sample training function (should be run first)
def train_model():
    try:
        # Load your dataset
        logger.info("Loading dataset...")
        df = pd.read_csv("data/emails.csv")  # Update path
        
        # Preprocess data
        X = df['masked_email'].tolist()  # Use masked emails for training
        y = df['category'].tolist()      # Email categories
        
        # Convert string labels to numerical
        label_map = {cat: i for i, cat in enumerate(set(y))}
        y_encoded = [label_map[label] for label in y]
        
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
        model.to(device)
        
        training_args = {
            'batch_size': 16,
            'epochs': 3,
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
            
            # Save training metrics
            logger.info(f"Epoch {epoch+1}/{training_args['epochs']}")
            logger.info(f"Train Loss: {train_loss/len(train_loader):.4f}")
            logger.info(f"Val Loss: {val_loss/len(val_loader):.4f}")
            logger.info(f"Val Accuracy: {correct/total:.4f}")
        
        # Create model directory if it doesn't exist
        if not os.path.exists("./model"):
            os.makedirs("./model")
            
        # Save model and tokenizer
        logger.info("Saving model...")
        model.save_pretrained("./model")
        tokenizer.save_pretrained("./model")
        
        # Save label mapping
        pd.DataFrame([label_map]).to_json("./model/label_map.json")
        
        logger.info("Training completed successfully!")
        return True
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return False

# Classification function
def classify_email(text: str):
    try:
        if not os.path.exists("./model"):
            logger.warning("Model directory not found. Using fallback classification.")
            return "general_inquiry"  # Fallback category
            
        tokenizer = BertTokenizer.from_pretrained("./model")
        model = BertForSequenceClassification.from_pretrained("./model")
        
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs).item()
        
        # Map numerical class to category name
        try:
            label_map = pd.read_json("./model/label_map.json").to_dict()
            inv_label_map = {v: k for k, v in label_map[0].items()}
            return inv_label_map.get(predicted_class, "general_inquiry")
        except:
            # If label map is missing, use default map
            return CATEGORY_MAP.get(predicted_class, "general_inquiry")
    
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return "uncategorized"
    
