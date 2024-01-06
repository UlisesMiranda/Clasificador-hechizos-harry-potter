import numpy as np
import librosa
import tensorflow as tf
import pyaudio
import wave

# Cargar el modelo entrenado
model = tf.keras.models.load_model('modelosAprendidos/pruebaCNN')

# Preprocesamiento de audio en tiempo real
def preprocess_audio_real_time(audio, sr=22050):
    duration = 2  # Duración deseada del audio en segundos
    # Si el audio es más largo que la duración especificada, se toma la parte central
    if len(audio) > sr * duration:
        start = (len(audio) - sr * duration) // 2
        audio = audio[start:start + sr * duration]

    # Realizar preprocesamiento para obtener el espectrograma
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec

# Función para capturar y procesar audio en tiempo real
def capture_audio_and_predict():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 22050
    RECORD_SECONDS = 2

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

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
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

    # Preprocesar el audio en tiempo real
    processed_audio = preprocess_audio_real_time(audio_data)

    # Ajustar la forma del audio para que sea compatible con el modelo
    processed_audio = np.expand_dims(processed_audio, axis=0)
    processed_audio = np.expand_dims(processed_audio, axis=-1)

    # Realizar la predicción con el modelo cargado
    prediction = model.predict(processed_audio)
    
    # Obtener la clase predicha
    predicted_class = np.argmax(prediction)
    
    # Mapear la clase predicha a un hechizo específico según el orden de tus clases
    spell_classes = ['Crucio', 'Desmayo', 'Imperio', 'Lumus', 'Protego', 'Reducto']
    predicted_spell = spell_classes[predicted_class]

    print(f"Hechizo predicho: {predicted_spell}")

# Llamar a la función para capturar audio en tiempo real y hacer una predicción
capture_audio_and_predict()
