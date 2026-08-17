#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Run database migrations
# This command assumes your alembic.ini is accessible and configured.
echo "Running database migrations..."
cd app
alembic upgrade heads  # Changed from 'head' to 'heads' to support multiple migration heads
cd ..

# Temporary workaround for adding release info
if ls docs/release_notes/*.yaml 1> /dev/null 2>&1; then
    for filename in docs/release_notes/*.yaml; do
        python tools/add_release_doc.py "$filename"
    done
else
    echo "No release notes found in docs/release_notes/"
fi

# Now, execute the command passed to the script (the uvicorn server command)
# The "exec" command replaces the script process with the new process,
# which is important for proper signal handling (e.g., Ctrl+C or docker stop).
exec "$@"
