import tkinter as tk
import threading
from mGrabaciones import grabar_audio_dir
from pruebaAudiosTest import predict_audio
import os

class ReconocimientoVozApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Modelo de reconocimiento de voz para clasificación de hechizos de Harry Potter")

       
        self.master.geometry("500x600")

        
        self.titulo_label = tk.Label(master, text="Modelo de reconocimiento de voz para clasificación de hechizos de Harry Potter", font=("Helvetica", 16))
        self.titulo_label.pack(pady=10)

       
        self.instrucciones_label = tk.Label(master, text="Presiona el botón azul para iniciar la grabación y el botón rojo para parar la grabación")
        self.instrucciones_label.pack()

        # Contenedor para los botones
        self.botones_frame = tk.Frame(master)
        self.botones_frame.pack(pady=10)

        # Botones de grabación y detención
        self.iniciar_button = tk.Button(self.botones_frame, text="Iniciar Grabación", command=self.iniciar_grabacion, bg="blue", fg="white", compound=tk.LEFT)
        self.iniciar_button.pack(side=tk.LEFT, padx=10)

        self.detener_button = tk.Button(self.botones_frame, text="Detener Grabación", command=self.detener_grabacion, bg="red", fg="black", compound=tk.LEFT, state=tk.DISABLED)
        self.detener_button.pack(side=tk.LEFT, padx=10)

        # Separador visual
        tk.Frame(master, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, padx=5, pady=5)

        # Campo de texto para mostrar hechizos detectados
        self.resultado_text = tk.Text(master, height=20, width=50)
        self.resultado_text.pack(pady=10)

        # Inicializar el hilo de grabación
        self.thread = None

    def iniciar_grabacion(self):
        self.resultado_text.delete(1.0, tk.END)  # Limpiar el campo de texto}
        self.grabar_audio()
        # self.iniciar_button.config(state=tk.DISABLED)
        # self.detener_button.config(state=tk.NORMAL)

        # Iniciar el hilo de grabación o mandar a reconocer el audio ya, a lo mejor cambiar a ponerlo cuando
        #le demos detener, esto sería mas para tiempo real
        # self.thread = threading.Thread(target=self.grabar_audio)
        # self.thread.start()

    def detener_grabacion(self):
        self.iniciar_button.config(state=tk.NORMAL)
        self.detener_button.config(state=tk.DISABLED)

        # # Detener el hilo de grabación
        # if self.thread and self.thread.is_alive():
        #     self.thread.join()

    def grabar_audio(self):
        #Aqui mandar a llamar el reconocedor
        audio_file = grabar_audio_dir("pruebas_audios")
        hechizo = predict_audio(audio_file)
        # Mostrar el hechizo detectado en el campo de texto
        self.resultado_text.insert(tk.END, f"Hechizo: {hechizo}\n")
        self.resultado_text.see(tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = ReconocimientoVozApp(root)
    root.mainloop()
    carpeta_a_borrar = 'pruebas_audios'

    # Eliminar todos los archivos de la carpeta
    for archivo in os.listdir(carpeta_a_borrar):
        ruta_completa = os.path.join(carpeta_a_borrar, archivo)
        if os.path.isfile(ruta_completa):
            os.remove(ruta_completa)
