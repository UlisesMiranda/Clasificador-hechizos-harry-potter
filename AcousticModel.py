from FeatureVector import FeatureVector
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier


class AcousticModel:
    """Class responsible to get the most probable word given a observation"""

    def __init__(self, n_classes: int):
        self.n_classes = n_classes

    def fit(self, feature_vectors: list[FeatureVector]):
        """Trains a K-Means model"""
        model = KMeans(self.n_classes)
        model.fit(X=feature_vectors)
        self.clusters = model.cluster_centers_
        self.predictor = KNeighborsClassifier(1)
        self.predictor.fit(self.clusters, range(self.n_classes))

    def predict(self, feature_vector: FeatureVector):
        """Predicts the class given a feature vector. Requires that the codebook has been previously fitted."""
        return self.predictor.predict(feature_vector)
