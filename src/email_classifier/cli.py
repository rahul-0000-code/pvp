"""Command-line interface for the email classifier package."""
import argparse
import logging
import sys
import os
from .models.classifier import train_model, classify_email
from .utils.pii_masker import mask_pii

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Email Classification CLI")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the classification model")
    train_parser.add_argument("--data", type=str, default="data/emails.csv", 
                            help="Path to the training dataset CSV")
    train_parser.add_argument("--epochs", type=int, default=3, 
                            help="Number of training epochs")
    
    # Classify command
    classify_parser = subparsers.add_parser("classify", help="Classify an email")
    classify_parser.add_argument("--email", type=str, required=True, 
                               help="Email text to classify")
    classify_parser.add_argument("--mask", action="store_true", 
                               help="Mask PII in the email")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start the API server")
    server_parser.add_argument("--host", type=str, default="0.0.0.0", 
                              help="Host IP address")
    server_parser.add_argument("--port", type=int, default=8000, 
                              help="Port number")
    
    # UI command
    ui_parser = subparsers.add_parser("ui", help="Start the Streamlit UI")
    
    return parser.parse_args()

def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    if args.command == "train":
        logger.info(f"Training model with data from {args.data} for {args.epochs} epochs")
        success = train_model(args.data, args.epochs)
        if success:
            logger.info("Training completed successfully")
        else:
            logger.error("Training failed")
            sys.exit(1)
    
    elif args.command == "classify":
        if args.mask:
            masked_email, entities = mask_pii(args.email)
            category = classify_email(masked_email)
            print(f"Original: {args.email}")
            print(f"Masked: {masked_email}")
            print(f"Entities: {entities}")
            print(f"Category: {category}")
        else:
            category = classify_email(args.email)
            print(f"Category: {category}")
    
    elif args.command == "server":
        os.environ["HOST"] = args.host
        os.environ["PORT"] = str(args.port)
        from .api.app import start
        logger.info(f"Starting API server on {args.host}:{args.port}")
        start()
    
    elif args.command == "ui":
        logger.info("Starting Streamlit UI")
        os.system("streamlit run src/email_classifier/ui/streamlit_app.py")
    
    else:
        print("No command specified. Use -h for help.")
        sys.exit(1)

if __name__ == "__main__":
    main() 