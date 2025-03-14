import sys
from pathlib import Path

from k_means_klang.ml_logic.data import load_data
from k_means_klang.ml_logic.preprocessor import preprocess_data
from k_means_klang.ml_logic.model import apply_pca, cluster_data

def classify_audio(csv_path: Path):
    """
    Classifies an audio feature CSV file into one of the trained clusters

    Returns:
    list: Predicted cluster labels
    """
    # Load data
    df = load_data(csv_path)

    # Preprocess data
    X_scaled, scaler = preprocess_data(df)

    # Apply PCA transformation
    X_pca, pca = apply_pca(X_scaled)

    # Cluster the transformed data
    cluster_labels = cluster_data(X_pca)

    return cluster_labels

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    # if not csv_path.is_file():
    #     raise ValueError(f"The file at {csv_path} does not exist or is not a valid file.")

    try:
        result = classify_audio(csv_path)
        print(f"Predicted Clusters: {result}")
    except Exception as e:
        print(f"Error: {e}")
