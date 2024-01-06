import os
import scipy
import scipy.io.wavfile as wav
import scipy.fft as fft
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from AcousticFrontend import AcousticFrontend
from AcousticModel import AcousticModel


def load_data(data_dir="./audios/", max_length_time=100, max_length_freq=128):
    X = []
    y = []

    labels = sorted(os.listdir(data_dir))

    for label_id, label in enumerate(labels):
        label_dir = os.path.join(data_dir, label)
        files = os.listdir(label_dir)
        np.random.shuffle(files)

        # split_index = int(0.8 * len(files))
        # train_files = files[:split_index]
        # test_files.extend(files[split_index:])

        for audio_file in files[:10]:
            audio_path = os.path.join(label_dir, audio_file)
            rate, audio = wav.read(audio_path)
            # audio = audio.astype(np.float32)  # Convertir a float32 para normalización
            # only saves the left channel
            X.append((audio[:, 0], rate))
            y.append(label_id)

    return X, y


def train(audios, classes):
    # Create feature vectors
    feature_vectors = []
    for i, audio in enumerate(audios):
        transformed = AcousticFrontend.transform(audio[0], audio[1], classes[i])
        feature_vectors.extend(transformed)

    # Fit the acoustic model
    n_spells = 6
    am = AcousticModel(n_spells)
    am.fit(feature_vectors)
    return am


if __name__ == "__main__":
    # Import all the audios
    audios = []
    classes = []
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the acoustic model
    am = train(X_train, y_train)
    # Preprocess the audio to test
    example_test, sample_rate_example = X_test[0]
    # left_channel_audio = example_test[:, 0]
    transformed = AcousticFrontend.transform(
        example_test, sample_rate_example, classification=y_test[0]
    )
    # predict the word
    print(am.predict(transformed))
