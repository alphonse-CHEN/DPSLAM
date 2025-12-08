#!/bin/bash

# Install CUDA toolkit
echo "Installing CUDA 12.6 toolkit..."
micromamba install -y -c nvidia/label/cuda-12.6.0 cuda-toolkit

# Install PyTorch with CUDA 12.6
echo "Installing PyTorch 2.7.0..."
pip3 install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126

# Install xformers
echo "Installing xformers..."
pip3 install -U xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu126

# Install PyTorch Geometric and related packages
echo "Installing PyTorch Geometric dependencies..."
pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.7.0+cu126.html

# Install PyTorch Geometric
pip3 install torch-geometric

# Install tensordict
echo "Installing tensordict..."
pip3 install tensordict
