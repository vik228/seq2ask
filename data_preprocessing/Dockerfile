# Use a smaller base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
# Combine apt-get update with apt-get install in the same RUN statement
# Clean up in the same RUN to reduce layer size
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends vim gcc python3-dev libpq-dev git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy your application code into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Further commands...