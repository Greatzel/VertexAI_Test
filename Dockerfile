# Base image
FROM python:3.8-slim-buster

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy pipeline code
COPY . .

# Set entrypoint
ENTRYPOINT ["python", "main_pipeline.py"]
