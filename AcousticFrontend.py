import numpy as np

from FeatureVector import FeatureVector


class AcousticFrontend:
    """Class responsible for transforming an audio into its corresponding feature vector"""

    def __init__(self):
        pass

    @staticmethod
    def transform(audio: np.ndarray, classification: str) -> FeatureVector:
        """
        Method responsible for transforming an audio into a feature vector
        """
        return FeatureVector(np.array([]), classification)
