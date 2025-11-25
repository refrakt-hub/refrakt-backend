#!/bin/bash
set -e

# Create cache directories with proper permissions (run as root)
mkdir -p /app/backend/cache/assistant_index
chown -R refrakt:refrakt /app/backend/cache 2>/dev/null || true

# Switch to refrakt user and execute the main command
exec gosu refrakt "$@"

