#!/usr/bin/env python3
"""Script to train the email classification model."""
import sys
import os
import argparse
import logging
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def check_dataset(data_path):
    """Check the dataset for required columns or suggest alternatives."""
    try:
        if not os.path.exists(data_path):
            logger.error(f"Dataset file not found: {data_path}")
            return False
            
        df = pd.read_csv(data_path)
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Dataset columns: {df.columns.tolist()}")
        
        # Check if required columns exist
        if 'masked_email' not in df.columns or 'category' not in df.columns:
            available_columns = df.columns.tolist()
            logger.warning(f"Required columns 'masked_email' or 'category' not found in dataset.")
            logger.info(f"Available columns: {available_columns}")
            
            # Try to identify suitable columns
            text_columns = [col for col in available_columns if any(term in col.lower() for term in ['text', 'email', 'content', 'body', 'message'])]
            label_columns = [col for col in available_columns if any(term in col.lower() for term in ['category', 'label', 'class', 'type'])]
            
            if text_columns and label_columns:
                logger.info(f"Suggested columns to use:")
                logger.info(f"  Text column: {text_columns[0]} (alternatives: {text_columns})")
                logger.info(f"  Label column: {label_columns[0]} (alternatives: {label_columns})")
                return True
            else:
                logger.error("Cannot find suitable text and label columns. Please check your dataset.")
                return False
        else:
            # Check class distribution
            categories = df['category'].value_counts()
            logger.info(f"Class distribution: \n{categories}")
            return True
    except Exception as e:
        logger.error(f"Error checking dataset: {e}")
        return False

def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(description="Train the email classification model")
    parser.add_argument("--data", type=str, default="data/emails.csv", 
                        help="Path to the training data CSV file")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--check-only", action="store_true",
                        help="Only check the dataset without training")
    args = parser.parse_args()
    
    # Check the dataset first
    logger.info(f"Checking dataset: {args.data}")
    if not check_dataset(args.data):
        logger.error("Dataset check failed. Please fix the issues before training.")
        return 1
    
    if args.check_only:
        logger.info("Dataset check completed. Skipping training as requested.")
        return 0
    
    try:
        # Import the training function from the package
        from src.email_classifier.models.classifier import train_model
        
        logger.info(f"Starting training with data from {args.data} for {args.epochs} epochs")
        success = train_model(args.data, args.epochs)
        
        if success:
            logger.info("Training completed successfully")
            return 0
        else:
            logger.error("Training failed")
            return 1
            
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 