import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

def preprocess_data(df, scaler=None):
    """
    Prepares and scales features for PCA and kmeans-clustering.
    """
    X_features = df.drop(["label", "filename", "length"], axis=1, errors="ignore")

    if scaler is None:
        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_features), columns=X_features.columns)
    else:
        scaler = scaler
        X_scaled = pd.DataFrame(scaler.fit_transform(X_features), columns=X_features.columns)

    print(f"✅ Preprocessing Completed")
    return X_scaled, scaler
