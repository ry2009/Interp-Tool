#!/bin/bash

# Install Homebrew if not installed
if ! command -v brew &> /dev/null
then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Brew installs as specified
brew install pyenv cmake protobuf rust

# Download and install Miniforge for arm64
curl -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh -b -p $HOME/miniforge
source $HOME/miniforge/bin/activate

# Create and activate conda env
conda create -n tilde-cpu python=3.11.9 -y
conda activate tilde-cpu

# Install PyTorch (latest since 2.3.0 may not be available directly; adjust if needed)
pip install torch torchvision torchaudio

# Install other packages
pip install transformer-lens==1.9 streamlit plotly pandas

# Instructions
echo "Setup complete. To activate the environment, run: conda activate tilde-cpu"
echo "To run Streamlit app: streamlit run your_app.py" 