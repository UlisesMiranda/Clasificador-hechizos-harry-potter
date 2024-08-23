import os
import librosa
import numpy as np
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential, load_model
from keras.regularizers import l2
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
import datetime as dt

class ModeloReconocimientoVozHarryPotter:
    def __init__(self, data_dir = '', max_length_time=2, max_length_mfcc=128):
        self.data_dir = data_dir
        self.max_length_time = max_length_time
        self.max_length_mfcc = max_length_mfcc
        self.model = None

    def load_data(self, data_dir):
        self.data_dir = data_dir
        X = []
        y = []

        labels = self.get_labels()

        for label_id, label in enumerate(labels):
            label_dir = self.get_label_dir(label)
            files = self.get_audio_files(label_dir)

            for audio_file in files:
                audio_path = os.path.join(label_dir, audio_file)
                mfcc = self.preprocess_audio(audio_path)

                X.append(mfcc)
                y.append(label_id)

        X = np.array(X)
        y = np.array(y)

        return X, y

    def get_labels(self):
        return sorted(os.listdir(self.data_dir))

    def get_label_dir(self, label):
        return os.path.join(self.data_dir, label)

    def get_audio_files(self, label_dir):
        files = os.listdir(label_dir)
        np.random.shuffle(files)
        return files

    def read_audio_file(self, audio_path):
        rate, audio = wavfile.read(audio_path)
        return rate, audio[:, 0]

    def trim_audio(self, audio, rate):
        max_length = int(rate * self.max_length_time)
        if len(audio) > max_length:
            start = (len(audio) - max_length) // 2
            audio = audio[start : start + max_length]
        return audio.astype(np.float32)

    def extract_mfcc(self, audio, rate):
        mfcc = librosa.feature.mfcc(y=audio, sr=rate, n_mfcc=self.max_length_mfcc)
        return mfcc.astype(np.float32)

    def pad_mfcc(self, mfcc):
        if mfcc.shape[1] < self.max_length_mfcc:
            pad_width = self.max_length_mfcc - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, :self.max_length_mfcc]
        return mfcc
    
    def preprocess_audio(self, audio_file):
        rate, audio = self.read_audio_file(audio_file)
        audio = self.trim_audio(audio, rate)
        mfcc = self.extract_mfcc(audio, rate)
        mfcc = self.pad_mfcc(mfcc)
        return mfcc

    def build_model(self):
        self.model = Sequential(
            [
                Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 1)),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation="relu"),
                MaxPooling2D((2, 2)),
                Conv2D(128, (3, 3), activation="relu"),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(128, activation="relu", kernel_regularizer=l2(0.001)),
                Dropout(0.5),
                Dense(6, activation="softmax"),
            ]
        )

        self.model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

    def train(self, X_train, y_train, epochs=20, batch_size=32):
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    def evaluate(self, X_test, y_test):
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        print(f"Loss en datos de prueba: {test_loss}")
        print(f"Precisión en datos de prueba: {test_accuracy}")
        return test_loss, test_accuracy

    def save_model(self, filename):
        self.model.save(filename)
        print(f"Modelo guardado como {filename}")
        
    def load_trained_model(self, model_path):
        self.model = load_model(model_path)
        print(f"Modelo cargado desde {model_path}")

    def predict_audio(self, audio_file):
        processed_audio = self.preprocess_audio(audio_file)
        processed_audio = np.expand_dims(processed_audio, axis=0)  # Ajustar la forma

        prediction = self.model.predict(processed_audio)
        confidence = np.max(prediction) * 100

        predicted_class = np.argmax(prediction)

        spell_classes = ['Crucio', 'Desmayo', 'Imperio', 'Lumus', 'Protego', 'Reducto']
        predicted_spell = spell_classes[predicted_class]

        print(f"Hechizo predicho: {predicted_spell}")
        print(f"Porcentaje de confianza: {confidence:.2f}%")

        return predicted_spell


if __name__ == "__main__":
    modelo = ModeloReconocimientoVozHarryPotter()

    DATA_DIR = "../audios/"
    X, y = modelo.load_data(DATA_DIR)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=89
    )
    
    modelo.build_model()
    modelo.train(X_train, y_train, epochs=20, batch_size=32)
    modelo.evaluate(X_test, y_test)

    current_date_and_time = dt.datetime.now().strftime("%Y%m%d%H")
    
    NOMBRE_MODELO = f"../models/modelo_{current_date_and_time}.h5"
    modelo.save_model(NOMBRE_MODELO)

    modelo.load_trained_model(NOMBRE_MODELO)
    modelo.predict_audio("./pruebas_audios/prueba_desmayo_vic.wav")