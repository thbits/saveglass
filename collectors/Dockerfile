FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the collector script
COPY eks_cloudwatch_collector.py .

# Create output directory
RUN mkdir -p /output

# Set the default output directory to /output (mounted volume)
ENV OUTPUT_DIR=/output

# Create a non-root user for security
RUN useradd -m -u 1000 collector
RUN chown -R collector:collector /app /output
USER collector

# Set the entrypoint
ENTRYPOINT ["python", "eks_cloudwatch_collector.py"]

# Default command (can be overridden)
CMD ["--help"]
