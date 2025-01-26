# Use the official Python 3.9 slim image as the base
FROM python:3.11-slim

# Install system dependencies for spaCy, scikit-learn, and other packages
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libpq-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libffi-dev \
    python3-dev \
    liblapack-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt into the container
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --verbose

# Download the spaCy model (en_core_web_sm)
RUN python -m spacy download en_core_web_sm || true  # The `|| true` ensures the build continues even if this step fails

# Copy the rest of the app files into the container
COPY app.py /app/app.py
COPY multi_label_model.pkl /app/multi_label_model.pkl
COPY vectorizer.pkl /app/vectorizer.pkl
COPY mlb.pkl /app/mlb.pkl
COPY domain_knowledge.json /app/domain_knowledge.json


# Expose port 5000 for the Flask app
EXPOSE 5000

# Command to run the Flask app when the container starts
CMD ["python", "app.py"]
