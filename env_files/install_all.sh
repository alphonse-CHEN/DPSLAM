#!/bin/bash

echo "Step 0: Cleaning previous build artifacts..."
bash env_files/00_clean_build.sh

echo "Step 1: Creating base environment with Python, GCC 13, and build tools..."
micromamba env create -f env_files/01_base_env.yml -y

echo "Step 2: Activating environment..."
eval "$(micromamba shell hook --shell bash)"
micromamba activate dpslam

echo "Step 3: Installing CUDA toolkit and PyTorch..."
bash env_files/02_install_pytorch.sh

echo "Step 4: Installing other dependencies..."
pip3 install -r env_files/03_requirements.txt

echo "Step 5: Installing DPSLAM package with CUDA extensions..."
pip3 install -e . --no-build-isolation

echo "✅ Installation complete! Activate with: micromamba activate dpslam"
