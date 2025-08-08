# K-Means Klang: Audio Clustering Project

**Live App:** [https://kmeansklang.streamlit.app/](https://kmeansklang.streamlit.app/)

This project was developed during Le Wagon's Data Science & AI Bootcamp (March 2025) by a team of 4. It clusters audio samples using K-Means based on acoustic features extracted with Librosa.

## Project Structure
- `Dockerfile`, `Makefile`, `requirements.txt` - Deployment/config
- `raw_data/` - Audio datasets
- `notebooks/` - Exploration notebooks
- `k_means_klang/` - Main code:
  - `api/fast.py` - FastAPI backend (predicts clusters for user-uploaded audio)
  - `interface/main.py` - Processes CSV files of pre-extracted audio features
  - `ml_logic/` - ML core (feature extraction, preprocessing, model)
  - `pickles/` - Saved models (K-Means, PCA, Scaler)

## How It Works
1. Extracts audio features using Librosa
2. Reduces dimensionality with PCA
3. Clusters samples using K-Means
4. Visualizes results in interactive Streamlit app

## Team
Zahra Mokhtari, Eleni Kartsiouka, Tai Ford, Luisa Freytag
