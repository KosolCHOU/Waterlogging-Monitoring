#!/usr/bin/env bash
# Render build script

set -e

echo "Installing dependencies..."
pip install -r requirements-minimal.txt

echo "Collecting static files..."
python manage.py collectstatic --noinput --clear

echo "Running migrations..."
python manage.py migrate --noinput

echo "Build completed!"
