#!/bin/bash
set -e

# Fix media directory permissions if needed
if [ -d "/app/media" ]; then
    echo "🔧 Ensuring media directory has correct permissions..."
    
    # Check if we have write access to media directory
    if [ ! -w "/app/media" ]; then
        echo "⚠️  Warning: Media directory not writable, attempting to fix..."
        # If running as root, fix permissions
        if [ "$(id -u)" = "0" ]; then
            chown -R app:app /app/media 2>/dev/null || true
        fi
    fi
    
    # Ensure subdirectories exist with proper permissions
    for dir in probes overlays hotspots insights plots stacks timeseries aoi avatars; do
        mkdir -p "/app/media/$dir"
        # Only change ownership if we're root
        if [ "$(id -u)" = "0" ]; then
            chown app:app "/app/media/$dir" 2>/dev/null || true
        fi
    done
fi

echo "🚀 Starting CropXcel application..."

# Execute the original command
exec "$@"