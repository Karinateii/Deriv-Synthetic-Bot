#!/bin/bash
# Build script for Render
# Upgrades pip and installs dependencies

set -e

echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Build complete!"
