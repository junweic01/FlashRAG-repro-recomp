#!/bin/bash
# GPU Setup Script for Amazon Linux 2023
# This script installs and configures NVIDIA drivers for GPU instances

set -e  # Exit on error

echo "=== Starting GPU Setup ==="

# Check if nvidia-smi is already working
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "GPU drivers already installed and working. Skipping installation."
    exit 0
fi

# Check if GPU hardware is present
if ! lspci | grep -i nvidia &> /dev/null; then
    echo "No NVIDIA GPU detected. Skipping GPU setup."
    exit 0
fi

echo "NVIDIA GPU detected. Installing drivers..."

# Add NVIDIA CUDA repository for Amazon Linux 2023
echo "Adding NVIDIA CUDA repository..."
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/amzn2023/x86_64/cuda-amzn2023.repo

# Clean metadata
sudo dnf clean all

# Enable NVIDIA driver module (using open-dkms which is the default)
echo "Enabling NVIDIA driver module..."
sudo dnf module enable -y nvidia-driver:open-dkms

# Install NVIDIA drivers and CUDA support
echo "Installing NVIDIA drivers and CUDA libraries..."
sudo dnf install -y nvidia-driver nvidia-driver-cuda nvidia-driver-cuda-libs

# Load NVIDIA kernel module
echo "Loading NVIDIA kernel module..."
sudo modprobe nvidia

# Verify installation
echo "Verifying GPU setup..."
if nvidia-smi; then
    echo "=== GPU Setup Complete! ==="
    nvidia-smi
else
    echo "ERROR: GPU setup failed. nvidia-smi is not working."
    exit 1
fi

