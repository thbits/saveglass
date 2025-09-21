# Use Python 3.13 slim image for smaller size
FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project configuration to leverage Docker cache
COPY pyproject.toml .

# Copy source code first for package installation
COPY src/ ./src/

# Copy resources directory for configuration and UI assets
COPY resources/ ./resources/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -e .

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose the port Streamlit runs on
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Command to run the application
CMD ["streamlit", "run", "src/agents_playground/app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true", "--server.fileWatcherType", "poll"]