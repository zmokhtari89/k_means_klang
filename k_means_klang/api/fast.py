import os
import shutil
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from typing import Dict
from k_means_klang.ml_logic.librosa import extract_features

app = FastAPI()

@app.get("/")
def root():
    return {'project':"k_means_klang"}

@app.post("/predict")
def predict(audio_file: UploadFile = File(...)):
    """
    Input is an audio_file: Uploaded audio file.
    Output is a ...?
    """
    # Save the uploaded file as a temporary file
    with open("temp_audio.wav", "wb") as buffer:  # Change the file extension to .wav or .mp3
        shutil.copyfileobj(audio_file.file, buffer)

    # Get the filepath to the saved temporary audio file
    temp_file_path = os.path.abspath("temp_audio.wav")  # Change to .wav, .mp3, or your preferred audio format

    # Call the make_prediction function on the temporary file using the temporary filepath
    features = extract_features(temp_file_path)
    features_df = pd.DataFrame(features, index = [0])
    X_features = features_df.drop(["label", "filename", "length"], axis=1, errors="ignore")

    # Load and apply scaler
    my_scaler = Path(__file__).resolve().parent.parent / "ml_logic" / "pickles" / "scaler.pkl"
    my_scaler=pickle.load(open(my_scaler, "rb"))

    X_new_scaled = my_scaler.transform(X_features)

    # Load and apply PCA
    pca = Path(__file__).resolve().parent.parent / "ml_logic" / "pickles" / "pca.pkl"
    pca = pickle.load(open(pca, 'rb'))
    X_new_pca = pca.transform(X_new_scaled)

    # Load and apply kmeans
    kmeans = Path(__file__).resolve().parent.parent / "ml_logic" / "pickles" / "kmeans.pkl"
    kmeans = pickle.load(open(kmeans, 'rb'))
    predictions = kmeans.predict(X_new_pca)

    return {"predicted_cluster": predictions.tolist()[0]}
