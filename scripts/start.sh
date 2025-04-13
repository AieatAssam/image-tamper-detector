#!/bin/bash

# Install backend dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install

# Start both frontend and backend
npm run dev & cd .. && uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000 