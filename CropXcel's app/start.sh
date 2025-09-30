#!/bin/bash
# Render startup script for CropXcel

set -e

echo "Starting CropXcel application..."
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"

# Start the Django application with Gunicorn
exec gunicorn cropxcel_project.wsgi:application --bind 0.0.0.0:$PORT --workers 2 --timeout 120
