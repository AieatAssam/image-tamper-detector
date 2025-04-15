#!/bin/bash

# Start the backend API in the background
cd /app && uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &

# Start Nginx in the foreground
nginx -g 'daemon off;' 