from FeatureVector import FeatureVector


class AcousticModel:
    """Class responsible to get the most probable word given a observation"""

    def __init__(self, n_classes: int):
        self.n_classes = n_classes

    def fit(self, feature_vectors: list[FeatureVector]):
        """Trains a K-Means model"""
        pass

    def predict(self, feature_vector: FeatureVector):
        """Predicts the class given a feature vector. Requires that the codebook has been previously fitted."""
        pass
