import numpy as np
import scipy.fft as fft

from FeatureVector import FeatureVector


class AcousticFrontend:
    """Class responsible for transforming an audio into its corresponding feature vector"""

    # def __init__(self):
    #     pass

    @staticmethod
    def transform(
        sampled_signal: np.ndarray, sample_rate, classification: str, w_size_ms=0.8
    ) -> FeatureVector:
        """
        Method responsible for transforming an audio into its feature vectors
        - sampled_signal: single channel of audio to transform
        - sample_rate: sample rate
        - classification: class of the audio (word)
        """
        window_size = int(sample_rate * w_size_ms)
        hop_size = window_size // 2
        spectrogram = AcousticFrontend._get_spectrogram(
            sampled_signal, window_size, hop_size
        )
        n_banks = 40
        n_ceps = 13
        # el numero de FFT's coincide con el tamaño de ventana
        mfcc = AcousticFrontend._get_mfcc(
            spectrogram, sample_rate, window_size, n_banks, n_ceps
        )
        feature_vectors = []
        for i in range(len(mfcc[0])):
            feature_vectors.append(FeatureVector(mfcc[:, i], classification))

        return feature_vectors

    def _stft(x, window_size, hop_size):
        """
        Devuelve la transformada para tiempo corto de Fourier
        y(k,m), donde
        y(k,m) es el valor de la transformada
        m el frame a consultar
        k es la frecuencia a obtener
        """
        # El numero de ventanas es seleccionado de modo que las ventanas estén completamente contenidas en el rango de tiempo de la señal
        # tambien se le conoce como numero de frames
        num_windows = (len(x) - window_size) // hop_size + 1
        stft = np.zeros((window_size, num_windows))
        for w in range(num_windows):
            start = w * hop_size
            end = start + window_size
            windowed_audio = x[start:end] * np.hamming(window_size)
            spectrum = np.fft.fft(windowed_audio)
            stft[:, w] = spectrum
        # limit the spectrum so returns only positive frequency part of the spectrum
        # Furthermore, K=N/2 (assuming that N is even) is the frequency index corresponding to the Nyquist frequency
        return stft[: 1 + window_size // 2]

    def _get_spectrogram(x, window_size, hop_size):
        stftx = AcousticFrontend._stft(x, window_size, hop_size)
        # el espectrograma puede ser definido como la STFT elevada al cuadrado
        spectrogram = np.abs(stftx) ** 2
        return spectrogram

    def _hz_to_mel(freq_hz):
        return 2595 * np.log10(1 + freq_hz / 700)

    def _mel_to_hz(freq_mel):
        return 700 * (10 ** (freq_mel / 2595) - 1)

    def _get_mel_filters(n_banks, sample_rate, nfft):
        lower_freq = AcousticFrontend._hz_to_mel(300)
        upper_freq = AcousticFrontend._hz_to_mel(sample_rate / 2)
        # los bancos de mel deben estar espaciados de forma igual en escala de mel
        filter_points_mel = np.linspace(lower_freq, upper_freq, n_banks + 2)
        filter_points_hz = AcousticFrontend._mel_to_hz(filter_points_mel)
        # a continuación determinamos que indices/valores de la FFT se relacionan con las frecuencias
        fft_bins = np.floor((nfft + 1) * filter_points_hz / sample_rate)
        filter_banks = np.zeros((n_banks, int(np.floor(nfft / 2 + 1))))

        for m in range(1, n_banks + 1):
            f_m_minus = int(fft_bins[m - 1])
            f_m = int(fft_bins[m])
            f_m_plus = int(fft_bins[m + 1])
            # Caso 2 de la formula
            for k in range(f_m_minus, f_m):
                filter_banks[m - 1, k] = (k - fft_bins[m - 1]) / (
                    fft_bins[m] - fft_bins[m - 1]
                )
            # Caso 3 de la formula
            for k in range(f_m, f_m_plus):
                filter_banks[m - 1, k] = (fft_bins[m + 1] - k) / (
                    fft_bins[m + 1] - fft_bins[m]
                )

        return filter_banks

    def _get_mfcc(power_spectrum, sample_rate, nfft, n_banks, n_ceps):
        """Regresa los MFCC en la forma [frame, coeficiente]"""
        filters = AcousticFrontend._get_mel_filters(n_banks, sample_rate, nfft)
        spectrum = np.dot(filters, power_spectrum)
        log_spectrum = np.log10(spectrum)
        mfcc = fft.dct(log_spectrum, type=2, axis=0, norm="ortho")[:(n_ceps), :]
        return mfcc
