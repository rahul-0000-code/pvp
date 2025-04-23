"""Setup script for the email_classifier package."""
from setuptools import setup, find_packages

# Define dependencies directly instead of reading from requirements.txt
install_requires = [
    "fastapi==0.109.2",
    "uvicorn==0.27.1",
    "spacy==3.7.4",
    "regex==2024.4.16",
    "torch==2.2.2",
    "transformers==4.39.3",
    "scikit-learn==1.4.2",
    "joblib==1.3.2",
    "pandas==2.2.2",
    "python-dateutil==2.9.0",
    "python-multipart==0.0.7",
    "streamlit==1.32.0",
    "accelerate>=0.21.0",
    "setuptools>=68.0.0",
    "wheel>=0.42.0",
]

setup(
    name="email_classifier",
    version="1.0.0",
    description="Email classification system with PII masking",
    author="Akaike Tech",
    author_email="info@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "email-classifier=email_classifier.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)