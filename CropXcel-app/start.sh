#!/bin/bash
# Startup script for CropXcel (works locally and on Render)

set -euo pipefail

PORT="${PORT:-8000}"
WORKERS="${WEB_CONCURRENCY:-1}"

echo "Starting CropXcel application..."
echo "Current directory: $(pwd)"
python --version

echo "Applying migrations..."
python manage.py migrate --noinput

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Starting Gunicorn on port ${PORT} with ${WORKERS} workers"
exec gunicorn cropxcel_project.wsgi:application \
  --bind 0.0.0.0:${PORT} \
  --workers ${WORKERS} \
  --timeout 120 \
  --log-level info