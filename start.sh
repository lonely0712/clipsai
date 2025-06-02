#!/bin/bash
# Script to start the FastAPI application with proper port handling

# Use PORT environment variable if set, otherwise default to 8000
PORT="${PORT:-8000}"

# Start the application
exec uvicorn app:app --host 0.0.0.0 --port "$PORT"
