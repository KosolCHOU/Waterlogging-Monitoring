#!/bin/bash
# Auto-fix probe file permissions - run this script periodically
# Usage: ./auto_fix_permissions.sh

echo "🔧 Auto-fixing media file permissions..."

# Get the current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIA_DIR="$SCRIPT_DIR/media"

if [ ! -d "$MEDIA_DIR" ]; then
    echo "❌ Media directory not found: $MEDIA_DIR"
    exit 1
fi

# Count files with wrong ownership
WRONG_OWNER_COUNT=$(find "$MEDIA_DIR" -user root 2>/dev/null | wc -l)

if [ "$WRONG_OWNER_COUNT" -gt 0 ]; then
    echo "📊 Found $WRONG_OWNER_COUNT files with root ownership"
    echo "🔧 Fixing ownership..."
    
    # Fix ownership of all root-owned files
    sudo chown -R kosol:kosol "$MEDIA_DIR"
    
    echo "✅ Fixed ownership of all media files"
else
    echo "✅ All media files have correct ownership"
fi

# Ensure proper permissions
chmod -R 755 "$MEDIA_DIR"
find "$MEDIA_DIR" -type f -exec chmod 644 {} \;

echo "✅ Media file permissions are now correct!"
echo "📁 Media directory: $MEDIA_DIR"
echo "👤 Owner: $(stat -c '%U:%G' "$MEDIA_DIR")"