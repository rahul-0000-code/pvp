#!/usr/bin/env python3
"""Script to train the model with the specific dataset structure."""
import sys
import os
import argparse
import logging
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import json
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default model directory
MODEL_DIR = "./model"

# Handle SSL certificate issues on macOS
if platform.system() == 'Darwin':
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    logger.info("SSL certificate verification disabled for macOS")
    
    # Also set environment variable for requests lib
    os.environ['CURL_CA_BUNDLE'] = ''
    
    # Try to fix HuggingFace Hub certificate issue
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    try:
        from huggingface_hub import constants
        # Force disable SSL verification in Hugging Face hub
        constants.HF_HUB_DISABLE_SYMLINKS_WARNING = True
        constants.HF_HUB_DISABLE_IMPLICIT_TOKEN = True
        constants.HF_HUB_DISABLE_PROGRESS_BARS = True
        constants.HF_HUB_DISABLE_TELEMETRY = True
    except:
        pass

def train_with_dataset(data_path, epochs=3, model_dir=MODEL_DIR):
    """Train a model using the actual dataset structure."""
    try:
        # Load dataset
        logger.info(f"Loading dataset from {data_path}...")
        df = pd.read_csv(data_path)
        logger.info(f"Dataset shape: {df.shape}")
        
        # Verify columns
        if 'email' not in df.columns or 'type' not in df.columns:
            logger.error(f"Required columns 'email' and 'type' not found in dataset!")
            return False
            
        # Map column names to expected names
        X = df['email'].tolist()   # Email text
        y = df['type'].tolist()    # Classification type
        
        # Show distribution
        class_counts = {}
        for label in y:
            class_counts[label] = class_counts.get(label, 0) + 1
            
        logger.info(f"Class distribution: {class_counts}")
        
        # Convert labels to numerical values
        label_map = {cat: i for i, cat in enumerate(set(y))}
        y_encoded = [label_map[label] for label in y]
        
        logger.info(f"Label mapping: {label_map}")
        
        # Save label mapping first in case of error
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        with open(os.path.join(model_dir, "label_map.json"), 'w') as f:
            json.dump({str(v): k for k, v in label_map.items()}, f)
        logger.info(f"Saved label mapping to {model_dir}/label_map.json")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        
        # Initialize model and tokenizer
        try:
            logger.info("Initializing BERT model...")
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
            model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased", 
                num_labels=len(label_map),
                local_files_only=True
            )
        except Exception as e:
            logger.warning(f"Could not load model with local_files_only=True: {e}")
            logger.info("Trying to download the model...")
            try:
                tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=False)
                model = BertForSequenceClassification.from_pretrained(
                    "bert-base-uncased", 
                    num_labels=len(label_map),
                    local_files_only=False
                )
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                logger.info("Creating a simple model instead...")
                # Create a simple classifier
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.linear_model import LogisticRegression
                
                # Train a TF-IDF + LogisticRegression model
                logger.info("Training a TF-IDF + LogisticRegression model...")
                vectorizer = TfidfVectorizer(max_features=5000)
                X_train_tfidf = vectorizer.fit_transform(X_train)
                X_val_tfidf = vectorizer.transform(X_val)
                
                clf = LogisticRegression(max_iter=1000)
                clf.fit(X_train_tfidf, y_train)
                
                # Save the scikit-learn model
                import joblib
                joblib.dump(vectorizer, os.path.join(model_dir, "vectorizer.joblib"))
                joblib.dump(clf, os.path.join(model_dir, "classifier.joblib"))
                joblib.dump(label_map, os.path.join(model_dir, "sklearn_label_map.joblib"))
                
                # Evaluate
                accuracy = clf.score(X_val_tfidf, y_val)
                logger.info(f"Validation accuracy: {accuracy:.4f}")
                
                logger.info("Training completed successfully (scikit-learn model)!")
                return True
        
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
        
        # Training setup
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
            
            for i, batch in enumerate(train_loader):
                if i % 100 == 0:
                    logger.info(f"Epoch {epoch+1}, Batch {i}/{len(train_loader)}")
                    
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
        
        # Save model and tokenizer
        logger.info(f"Saving model to {model_dir}...")
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        
        logger.info("Training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(description="Train email classification model with specific dataset")
    parser.add_argument("--data", type=str, default="data/emails.csv", 
                        help="Path to the CSV dataset")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR,
                        help="Directory to save the model")
    parser.add_argument("--fallback", action="store_true",
                        help="Use sklearn fallback model instead of BERT")
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        logger.error(f"Dataset file not found: {args.data}")
        return 1
    
    success = train_with_dataset(args.data, args.epochs, args.model_dir)
    
    if success:
        logger.info("Training completed successfully")
        return 0
    else:
        logger.error("Training failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 