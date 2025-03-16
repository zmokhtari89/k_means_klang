# import os
# import shutil
# from sklearn.preprocessing import MinMaxScaler

# from fastapi import FastAPI, File, UploadFile
# from typing import Dict
# from k_means_klang.ml_logic.librosa import extract_features
# from k_means_klang.ml_logic.preprocessor import preprocess_data
# from k_means_klang.ml_logic.model import cluster_data

# app = FastAPI()

# @app.get("/")
# def root():
#     return {'project':"k_means_klang"}

# @app.post("/predict")
# def predict(audio_file: UploadFile = File(...)):
#     """
#     Input is an audio_file: Uploaded audio file.
#     Output is a ...?
#     """
#     # Save the uploaded file as a temporary file
#     with open("temp_audio.wav", "wb") as buffer:  # Change the file extension to .wav or .mp3
#         shutil.copyfileobj(audio_file.file, buffer)

#     # Get the filepath to the saved temporary audio file
#     temp_file_path = os.path.abspath("temp_audio.wav")  # Change to .wav, .mp3, or your preferred audio format

#     # Call the make_prediction function on the temporary file using the temporary filepath
#     features = extract_features(temp_file_path)
#     processed_features = preprocess_data(features, scaler = MinMaxScaler())
#     predictions = cluster_data(processed_features)

#     # Remove the temporary file
#     os.remove(temp_file_path)

#     return predictions


import os
import shutil
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List
import librosa
import pandas as pd

from k_means_klang.ml_logic.librosa import extract_features
from k_means_klang.ml_logic.preprocessor import preprocess_data
from k_means_klang.ml_logic.model import cluster_data

app = FastAPI()

@app.get("/")
def root():
    return {"project": "k_means_klang"}

@app.post("/predict")
async def predict(audio_file: UploadFile = File(...)):
    """
    Accepts an uploaded audio file and returns a prediction.
    """

    # Ensure the file has an acceptable extension
    allowed_extensions = {".wav", ".mp3", ".flac"}
    file_ext = os.path.splitext(audio_file.filename)[-1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_ext}. Use .wav, .mp3, or .flac")

    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_audio:
        shutil.copyfileobj(audio_file.file, temp_audio)
        temp_file_path = temp_audio.name  # Full path of temp file

    try:
        # Verify the file is a valid audio file before processing
        try:
            librosa.load(temp_file_path, sr=None)  # Check if it's readable
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid or corrupt audio file: {str(e)}")

        # Extract features
        features = extract_features(temp_file_path)

        if features is None:
            raise HTTPException(status_code=500, detail="Feature extraction failed. Invalid audio data.")

        features_df = pd.DataFrame(features, index = [0])

        print(features_df.shape)
        # Preprocess features
        processed_features, scalar = preprocess_data(features_df)

        print(type(processed_features))

        # Get predictions
        predictions = cluster_data(processed_features, n_clusters=1)

        # Ensure the response is JSON serializable
        return {"predicted_cluster": predictions.tolist() if hasattr(predictions, "tolist") else predictions}

    finally:
        # Cleanup temp file to prevent unnecessary storage usage
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
