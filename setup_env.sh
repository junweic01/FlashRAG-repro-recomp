#!/bin/bash
sudo dnf install -y htop

# Setup GPU drivers if GPU is present
chmod +x ./setup_gpu.sh
./setup_gpu.sh

# Install Miniconda only if not already installed
if [ ! -d "$HOME/miniconda3" ]; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b
else
    echo "Miniconda already installed. Skipping."
fi


export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

if ! conda info --envs | grep -q "flashrag"; then
    conda create -n flashrag python=3.10 -y
fi
conda activate flashrag

pip install -e . 

# Fix dependency compatibility issues
pip install "accelerate<1.0.0" "scipy<1.15" --upgrade
pip install termcolor
conda install -c pytorch faiss-gpu -y
pip install vllm
