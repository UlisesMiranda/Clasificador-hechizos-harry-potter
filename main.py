import scipy
import scipy.io.wavfile as wav
import scipy.fft as fft
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt

from AcousticFrontend import AcousticFrontend
from AcousticModel import AcousticModel


def train(audios, classes):
    # Create feature vectors
    feature_vectors = []
    for i, audio in enumerate(audios):
        transformed = AcousticFrontend.transform(audio, classes[i])
        feature_vectors.append(transformed)

    # Fit the acoustic model
    n_spells = 6
    am = AcousticModel(n_spells)
    am.fit(feature_vectors)
    return am


if __name__ == "__main__":
    # Import all the audios
    audios = []
    classes = []

    # Train the acoustic model
    am = train(audios, classes)
    # Preprocess the audio to test
    transformed = None
    # predict the word
    am.predict(transformed)
