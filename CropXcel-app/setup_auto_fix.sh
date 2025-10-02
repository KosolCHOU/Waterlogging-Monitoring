#!/bin/bash
# Setup cron job to automatically fix media file permissions
# Run this script once to install the cron job

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRON_CMD="*/5 * * * * $SCRIPT_DIR/auto_fix_permissions.sh >/dev/null 2>&1"

echo "📅 Setting up automatic permission fixing every 5 minutes..."

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "auto_fix_permissions.sh"; then
    echo "✅ Cron job already exists"
else
    # Add cron job
    (crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -
    echo "✅ Cron job installed successfully!"
fi

echo "📋 Current cron jobs:"
crontab -l 2>/dev/null | grep -E "(auto_fix_permissions|media)" || echo "No relevant cron jobs found"

echo ""
echo "🔧 To manually fix permissions now, run:"
echo "  ./auto_fix_permissions.sh"
echo ""
echo "📅 Automatic fixing is now scheduled every 5 minutes"
echo "🗑️  To remove the cron job later, run:"
echo "  crontab -e"