import librosa
import os
import numpy as np
from scipy.io import wavfile
from keras.models import load_model

# Cargar el modelo entrenado
model = load_model('modeloPrueba.h5')

def preprocess_audio(audio_file, max_length_time=2, max_length_mfcc=128):
    rate, audio = wavfile.read(audio_file)
    audio = audio[:, 0]

    # Limitar la longitud de la señal al máximo deseado (2 segundos en este caso)
    max_length = int(rate * max_length_time)
    if len(audio) > max_length:
        start = (len(audio) - max_length) // 2
        audio = audio[start:start + max_length]
        
    audio = audio.astype(np.float32)  # Convertir a float32 para normalización
    audio /= np.max(np.abs(audio))  # Normalizar entre -1 y 1

    # Extraer coeficientes MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=rate, n_mfcc=max_length_mfcc)
    mfcc = mfcc.astype(np.float32)

    # Ajustar la longitud de los MFCC
    if mfcc.shape[1] < max_length_mfcc:
        pad_width = max_length_mfcc - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_length_mfcc]  # Acortar si es más largo

    return mfcc

def predict_audio(audio_file):
    processed_audio = preprocess_audio(audio_file)
    processed_audio = np.expand_dims(processed_audio, axis=0)  # Ajustar la forma
    
    # Realizar la predicción con el modelo cargado
    prediction = model.predict(processed_audio)
    confidence = np.max(prediction) * 100
    
    # Obtener la clase predicha
    predicted_class = np.argmax(prediction)
    
    # Mapear la clase predicha a un hechizo específico según el orden de tus clases
    spell_classes = ['Crucio', 'Desmayo', 'Imperio', 'Lumus', 'Protego', 'Reducto']
    predicted_spell = spell_classes[predicted_class]

    print(f"Hechizo predicho: {predicted_spell}")
    print(f"Porcentaje de confianza: {confidence:.2f}%")
    
    return predicted_spell

# # Obtener la lista de archivos en la carpeta raíz del proyecto
# audio_files = [file for file in os.listdir() if file.endswith('.wav')]

# # Iterar sobre cada archivo y hacer una predicción
# for audio_file in audio_files:
#     print(f"Prediciendo el archivo: {audio_file}")
#     predict_audio(audio_file)
#     print("=" * 50)
