FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and setup files first for better layer caching
COPY requirements.txt setup.py install_models.py ./

# Install dependencies (excluding the URL-based ones)
RUN pip install --no-cache-dir -r requirements.txt

# Install the package in development mode
RUN pip install -e .

# Install SpaCy language models
RUN python install_models.py

# Copy the rest of the code
COPY . .

# Create necessary directories
RUN mkdir -p model

# Set environment variables
ENV HF_SPACE_APP=api
ENV PORT=7860

# Expose port
EXPOSE 7860

# Start the application
CMD ["python", "app.py"] 