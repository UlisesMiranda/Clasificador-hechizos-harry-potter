import sounddevice as sd
from scipy.io.wavfile import write
import os
from datetime import datetime

def grabar_audio(ruta_carpeta, duracion_segundos=5, frecuencia_muestreo=44100):
    if not os.path.exists(ruta_carpeta):
        os.makedirs(ruta_carpeta)

    duracion_frames = int(duracion_segundos * frecuencia_muestreo)
    formato_audio = 'int16'

    print("Grabando audio...")
    audio = sd.rec(duracion_frames, samplerate=frecuencia_muestreo, channels=1, dtype=formato_audio)
    sd.wait()

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    nombre_archivo = f"{inicio}_{timestamp}.wav"

    ruta_archivo = os.path.join(ruta_carpeta, nombre_archivo)
    write(ruta_archivo, frecuencia_muestreo, audio)

    print(f"Audio guardado en: {ruta_archivo}")

ruta_carpeta_grabaciones = "grabaciones de prueba"
inicio = "L"
grabar_audio(ruta_carpeta_grabaciones)
