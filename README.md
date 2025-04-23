# Email Classification API

An API system that masks PII/PCI data in emails, classifies them into predefined categories, and returns the masked content along with PII metadata.

## Features

1. **PII/PCI Masking**

   - Detects and masks sensitive information like names, emails, phone numbers, credit card details, etc.
   - Uses regex patterns and SpaCy NER for detection without LLMs
2. **Email Classification**

   - Classifies masked emails into predefined categories
   - Uses BERT model for sequence classification
3. **API Implementation**

   - FastAPI-based RESTful API
   - Input: Email body text
   - Output: Masked email, original PII entities with positions, and email category

## Project Structure

```
email_classifier/
├── data/                       # Training and test data
├── model/                      # Saved model files
├── src/                        # Source code
│   └── email_classifier/       # Main package
│       ├── __init__.py
│       ├── cli.py              # Command-line interface
│       ├── api/                # API implementation
│       │   ├── __init__.py
│       │   ├── app.py          # FastAPI application
│       │   ├── api_models.py   # Pydantic models
│       │   └── routes.py       # API routes
│       ├── models/             # ML models
│       │   ├── __init__.py
│       │   └── classifier.py   # Email classification
│       ├── utils/              # Utilities
│       │   ├── __init__.py
│       │   └── pii_masker.py   # PII masking
│       └── ui/                 # User interfaces
│           └── streamlit_app.py  # Streamlit UI
├── main.py                     # Main entry point
├── setup.py                    # Package setup
└── requirements.txt            # Dependencies
```

## Installation & Setup

```bash
# Clone the repository
git clone [repository-url]

# Navigate to project directory
cd [project-directory]

# Install dependencies
pip install -r requirements.txt

# Install SpaCy language models
python install_models.py

# Install the package in development mode
pip install -e .
```

## Usage

### Command Line Interface

```bash
# Train the model
email-classifier train --data data/emails.csv --epochs 3

# Classify an email with PII masking
email-classifier classify --email "Hello, my name is John Doe" --mask

# Start the API server
email-classifier server --host 0.0.0.0 --port 8000

# Start the Streamlit UI
email-classifier ui
```

### Python API

```python
from email_classifier.utils.pii_masker import mask_pii
from email_classifier.models.classifier import classify_email

# Mask PII in text
masked_text, entities = mask_pii("Hello, my name is John Doe and my email is john@example.com")

# Classify email
category = classify_email(masked_text)

print(f"Masked: {masked_text}")
print(f"Entities: {entities}")
print(f"Category: {category}")
```

## API Usage

### POST /api/classify

**Request:**

```json
{
  "email_body": "Hello, my name is Prathmest Patil PE DS, and my email is johndoe@example.com."
}
```

**Response:**

```json
{
  "input_email_body": "Hello, my name is Prathmest Patil PE DS, and my email is johndoe@example.com.",
  "list_of_masked_entities": [
    {
      "position": [21, 29],
      "classification": "full_name",
      "entity": "Prathmest Patil PE DS"
    },
    {
      "position": [48, 70],
      "classification": "email",
      "entity": "prathemst@example.com"
    }
  ],
  "masked_email": "Hello, my name is [full_name], and my email is [email].",
  "category_of_the_email": "Account Management"
}
```

## Docker Deployment

You can use Docker to deploy the application:

```bash
# Build the Docker image
docker build -t email-classifier .

# Run the API server
docker run -p 7860:7860 email-classifier

# Run the Streamlit UI
docker run -p 7860:7860 -e HF_SPACE_APP=ui email-classifier
```

## Hugging Face Spaces Deployment

To deploy on Hugging Face Spaces:

1. Create a new Space on Hugging Face
2. Select "Streamlit" or "FastAPI" depending on your preference
3. Upload the code to the repository
4. Configure the environment variables (set HF_SPACE_APP to "api" or "ui")
5. Hugging Face will automatically build and deploy the application using the Dockerfile
