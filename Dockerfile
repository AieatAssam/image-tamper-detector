# Build stage for frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Build stage for backend
FROM python:3.11-slim AS backend-builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend dependencies and code
COPY --from=backend-builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY backend/ ./backend/

# Copy frontend build
COPY --from=frontend-builder /app/frontend/dist/ ./frontend/dist/

# Copy startup script
COPY scripts/start-docker.sh ./
RUN chmod +x start-docker.sh

# Install a lightweight web server for frontend
RUN pip install --no-cache-dir uvicorn

# Expose ports
EXPOSE 8000 5173

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Start both services
CMD ["./start-docker.sh"] 