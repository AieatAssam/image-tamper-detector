#!/bin/bash

# Start the frontend static file server
cd /app/frontend/dist && python -m http.server 5173 &

# Start the backend API
cd /app && uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 