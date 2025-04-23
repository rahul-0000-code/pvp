import subprocess
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Installing spaCy English language model (en_core_web_sm)...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl"
        ], check=True, capture_output=True, text=True)
        
        logger.info("spaCy model installation completed successfully!")
        return 0
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing spaCy model: {e}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        return 1
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 