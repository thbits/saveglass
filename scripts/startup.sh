#!/bin/bash
# Startup script for AI Agents Playground

echo "🚀 Starting AI Agents Playground..."

# Ensure required directories exist
mkdir -p /app/.auth /app/.sessions

# Ensure yaronG and admin users exist
echo "🔧 Setting up required users..."
python3 /app/scripts/ensure_yarong_user.py

# Start the Streamlit application
echo "🎯 Starting Streamlit application..."
exec streamlit run src/agents_playground/app.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true \
    --server.fileWatcherType poll
