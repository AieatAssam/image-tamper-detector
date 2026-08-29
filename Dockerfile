# Build stage for frontend
FROM node:24-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Final stage
FROM python:3.14-slim
WORKDIR /app

# Install system dependencies for OpenCV and Nginx
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx libcap2-bin \
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
RUN groupadd --system app && useradd --system --gid app --home-dir /app app \
    && mkdir -p /var/lib/nginx /var/log/nginx /var/cache/nginx /run \
    && chown -R app:app /var/lib/nginx /var/log/nginx /var/cache/nginx /run /app \
    && setcap cap_net_bind_service=+ep /usr/sbin/nginx \
    && chmod +x start-docker.sh

# Expose single port for Nginx
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1/healthz', timeout=3)"

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

USER app

# Start services
CMD ["./start-docker.sh"]
