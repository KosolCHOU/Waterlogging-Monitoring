#!/bin/bash
# Fix media file permissions script for CropXcel
# Run this if you encounter permission issues with probe files

echo "🔧 Fixing media file permissions..."

# Fix ownership of all media files
sudo chown -R kosol:kosol media/

# Make sure directories are readable/writable
chmod -R 755 media/

# Make sure files are readable
find media/ -type f -exec chmod 644 {} \;

echo "✅ Media file permissions fixed!"
echo "📁 Media directory structure:"
ls -la media/