#!/bin/bash

# Initialize Conda for the script
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create a new Conda environment with Python 3.10 specifically
conda create -n cscnns python=3.10 -y

# Activate the environment
conda activate cscnns

# Install lie-learn from the cloned repository
cd lie_learn
pip install .
cd ..

# Install PyTorch and related packages (for Apple Silicon)
conda install pytorch torchvision torchaudio -c pytorch -y

# Install missing dependencies
pip install torch-tb-profiler torchmetrics

# Install escnn (after PyTorch is installed)
pip install escnn -q --no-cache

# JAX installation for CPU
pip install --upgrade "jax[cpu]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install flax

# Install additional packages
pip install matplotlib wandb
pip install cliffordlayers
pip install neuraloperator

echo "Environment 'cscnns' created and packages installed successfully."


