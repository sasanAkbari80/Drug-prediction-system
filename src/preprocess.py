import numpy as np
from sklearn.preprocessing import LabelEncoder

def encode_features(X):
    X = X.copy()

    sex_encoder = LabelEncoder()
    X[:, 1] = sex_encoder.fit_transform(X[:, 1])

    bp_encoder = LabelEncoder()
    X[:, 2] = bp_encoder.fit_transform(X[:, 2])

    chol_encoder = LabelEncoder()
    X[:, 3] = chol_encoder.fit_transform(X[:, 3])

    return X
