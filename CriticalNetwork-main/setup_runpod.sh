#!/bin/bash
# RunPod setup script - run this once after creating your pod

set -e

echo "=== Installing system dependencies ==="
apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

echo "=== Installing Python dependencies ==="
pip install --no-cache-dir -r requirements.txt

echo "=== Setup complete ==="
echo "Run the simulation with: python main_simulation.py"
echo "Or run the simple example with: python simple_example.py"
