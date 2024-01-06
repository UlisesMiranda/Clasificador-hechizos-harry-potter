import numpy as np
import librosa
import tensorflow as tf
import pyaudio
import wave

# Cargar el modelo entrenado
model = tf.keras.models.load_model("modeloPrueba.h5")


# Preprocesamiento de audio en tiempo real
def preprocess_audio_real_time(audio, sr=22050, max_length_mfcc=128):
    duration = 2  # Duración deseada del audio en segundos
    # Si el audio es más largo que la duración especificada, se toma la parte central
    if len(audio) > sr * duration:
        start = (len(audio) - sr * duration) // 2
        audio = audio[start : start + sr * duration]

    audio = audio.astype(np.float32)  # Convertir a float32 para normalización
    audio /= np.max(np.abs(audio))  # Normalizar entre -1 y 1

    # Extraer coeficientes MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=max_length_mfcc)
    mfcc = mfcc.astype(np.float32)

    # Ajustar la longitud de los MFCC
    if mfcc.shape[1] < max_length_mfcc:
        pad_width = max_length_mfcc - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_length_mfcc]  # Acortar si es más largo
    return mfcc

    audio = audio.astype(np.float32)  # Convertir a float32 para normalización
    audio /= np.max(np.abs(audio))  # Normalizar entre -1 y 1

    # Extraer coeficientes MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=max_length_mfcc)
    mfcc = mfcc.astype(np.float32)

    # Ajustar la longitud de los MFCC
    if mfcc.shape[1] < max_length_mfcc:
        pad_width = max_length_mfcc - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_length_mfcc]  # Acortar si es más largo
    return mfcc


# Función para capturar y procesar audio en tiempo real
def capture_audio_and_predict():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 22050
    RECORD_SECONDS = 3

    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    print("* Comenzando la grabación...")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* Grabación finalizada")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Convertir los datos de audio capturados en un arreglo numpy
    audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)

    # Convertir audio a formato de punto flotante
    audio_data_float = librosa.util.buf_to_float(audio_data, dtype=np.float32)

    # Convertir audio a formato de punto flotante
    audio_data_float = librosa.util.buf_to_float(audio_data, dtype=np.float32)

    # Preprocesar el audio en tiempo real
    processed_audio = preprocess_audio_real_time(audio_data_float)

    # Ajustar la forma del audio para que sea compatible con el modelo
    processed_audio = np.expand_dims(processed_audio, axis=0)
    processed_audio = np.expand_dims(processed_audio, axis=-1)

    # Realizar la predicción con el modelo cargado
    prediction = model.predict(processed_audio)
    confidence = np.max(prediction) * 100

    print(prediction)

    # Obtener la clase predicha
    predicted_class = np.argmax(prediction)

    print(predicted_class)

    # Mapear la clase predicha a un hechizo específico según el orden de tus clases
    spell_classes = ["Crucio", "Desmayo", "Imperio", "Lumus", "Protego", "Reducto"]
    predicted_spell = spell_classes[predicted_class]

    print(f"Hechizo predicho: {predicted_spell}")
    print(f"Porcentaje de confianza: {confidence:.2f}%")


# Llamar a la función para capturar audio en tiempo real y hacer una predicción
capture_audio_and_predict()
