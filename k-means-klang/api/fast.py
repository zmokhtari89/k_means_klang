import os
import shutil
from fastapi import FastAPI, File, UploadFile
from typing import Dict
from k-means-klang.ml_logic.librosa import extract_features
from k-means-klang.ml_logic.preprocessor import preprocess_data
from k-means-klang.ml_logic.model import cluster_data

app = FastAPI()

@app.get("/")
def root():
    return {'greeting':"hello"}

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
    processed_features = preprocess_data(features)
    predictions = cluster_data(processed_features)

    # Remove the temporary file
    os.remove(temp_file_path)

    return predictions
