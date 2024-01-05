import numpy as np


class FeatureVector:
    """Class responsible for storing a feature vector with its class"""

    def __init__(self, feature_vector: np.ndarray, classification: str):
        self.feature_vector = feature_vector
        self.classification = classification
