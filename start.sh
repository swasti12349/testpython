#!/usr/bin/env bash
set -e

cd /app
# Start nginx in the foreground
nginx -g 'daemon off;' &

# Replace this shell with your Streamlit process
exec streamlit run auth.py \
     --server.address 0.0.0.0 \
     --server.port 8501