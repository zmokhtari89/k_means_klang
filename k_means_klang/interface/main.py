import sys
from data import load_data
from preprocessing import preprocess_data
from model import apply_pca, cluster_data

def classify_audio(csv_path):
    """
    Classifies an audio feature CSV file into one of the trained clusters

    Returns:
    list: Predicted cluster labels
    """
    # Load data
    df = load_data(csv_path)

    # Preprocess data
    X_scaled = preprocess_data(df)

    # Apply PCA transformation
    X_pca = apply_pca(X_scaled)

    # Cluster the transformed data
    cluster_labels = cluster_data(X_pca)

    return cluster_labels

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]

    try:
        result = classify_audio(csv_path)
        print(f"Predicted Clusters: {result}")
    except Exception as e:
        print(f"Error: {e}")
