# Build stage for frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Final stage
FROM python:3.12-slim
WORKDIR /app

# Install system dependencies for OpenCV and Nginx
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies and copy backend code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./backend/

# Copy frontend build
COPY --from=frontend-builder /app/frontend/dist/ ./frontend/dist/

# Copy Nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy startup script
COPY scripts/start-docker.sh ./
RUN chmod +x start-docker.sh

# Expose single port for Nginx
EXPOSE 80

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Start services
CMD ["./start-docker.sh"] 