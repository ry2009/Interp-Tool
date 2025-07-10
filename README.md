# Interp-Toolkit

A compute-light, publication-grade interpretability toolkit for analyzing frozen open-weight LLMs on macOS laptops.

## Setup

1. Run `./setup.sh` to install dependencies and set up the conda environment.

Alternatively, use `conda env create -f environment.yml` to create the environment.

2. Activate the environment: `conda activate tilde-cpu`

## Running the Toolkit

Run `streamlit run src/app.py` to launch the visualization dashboard.

## Case Study

Regex-sink mitigation using activation dumps from TinyLlama-1.1B and Gemma-2B-It.

## Deliverables

- Mini-paper in docs/
- Activation sample pack in samples/
- Colab notebook in notebooks/
- 6-week roadmap in docs/roadmap.md 