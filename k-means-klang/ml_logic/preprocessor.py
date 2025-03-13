import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    """
    prepares and scales features for PCA and kmeans-clustering.
    """
    X_features = df.drop(["label", "filename", "length"], axis=1)
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_features), columns=X_features.columns)
    return X_scaled

