#!/bin/bash

echo "Cleaning previous build artifacts..."

# Remove build directories
rm -rf build/
rm -rf dist/
rm -rf *.egg-info
rm -rf dpvo.egg-info

# Remove compiled Python files
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -delete

# Remove compiled CUDA/C++ extensions
find . -type f -name "*.so" -delete
find . -type f -name "*.o" -delete

echo "✅ Cleanup complete!"
