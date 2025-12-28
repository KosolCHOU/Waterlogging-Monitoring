#!/bin/sh

# Wait for redis if needed (optional since we use sqlite for DB, 
# but good for celery)

# Handle Earth Engine Credentials from Env Var (for Render)
if [ -n "$EE_CREDENTIALS" ]; then
    echo "Detected EE_CREDENTIALS, writing to config..."
    mkdir -p /root/.config/earthengine
    echo "$EE_CREDENTIALS" > /root/.config/earthengine/credentials
fi

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput

# Apply database migrations
echo "Applying database migrations..."
python manage.py migrate

# Start server
echo "Starting server..."
exec "$@"
