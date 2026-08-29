#!/usr/bin/env bash
set -Eeuo pipefail

cd /app
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &
backend_pid=$!
nginx -g 'daemon off;' &
nginx_pid=$!
trap 'kill "$backend_pid" "$nginx_pid" 2>/dev/null || true' EXIT
wait -n "$backend_pid" "$nginx_pid"
exit 1
