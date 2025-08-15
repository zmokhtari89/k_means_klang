K-Means Klang: Audio Clustering Project
Live App: https://kmeansklang.streamlit.app/

This project was developed during Le Wagon's Data Science & AI Bootcamp (March 2025) by a team of four. It clusters audio samples using K-Means based on acoustic features extracted with the Librosa library. The project uses a variety of audio features—such as Mel-frequency cepstral coefficients (MFCCs), spectral contrast, and harmonic features—to find hidden patterns in music beyond genre.

⚙️ How It Works
Feature Extraction: Extracts acoustic features from audio files using the Librosa library.

Dimensionality Reduction: Reduces the feature space using Principal Component Analysis (PCA).

Clustering: Applies the K-Means algorithm to group the audio samples into distinct clusters.

Visualization: Displays the results in an interactive Streamlit application.

🚀 Installation
Follow these steps to set up the project locally.

1. Download the Dataset
The project uses the GTZAN Dataset, which is not included in the repository due to its size.

Download Data.zip from this Kaggle page: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification.

Unzip Data.zip.

Place the unzipped Data folder inside the raw_data/ directory. Your folder structure should look like this: raw_data/Data/genres_original/.

2. Set Up the Environment
First, install the necessary system packages, then create and activate a Python virtual environment.

sudo apt-get install virtualenv python3-pip python3-dev
deactivate; virtualenv -p python3 ~/venv ; source ~/venv/bin/activate

3. Clone and Install
Clone the repository and install the required dependencies.

git clone git@github.com:zmokhtari89/k_means_klang.git
pip install -r requirements.txt
make clean install test

4. Run the Project
After installation, you can run the main script. The -m flag allows Python to run a module as a script, correctly handling relative imports within the project package.

python -m k_means_klang.interface.main

🤝 Team
Zahra Mokhtari

Eleni Kartsiouka

Tai Ford

Luisa Freytag
