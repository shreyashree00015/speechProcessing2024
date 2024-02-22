import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt

filename = "SP_shreya_audioFile.wav"
y, sr = librosa.load(filename)

#A1
a = np.fft.fft(y)
 
freq = np.fft.fftfreq(len(a), d=1/sr)

plt.figure(figsize=(10, 5))
plt.plot(freq, np.abs(a))
plt.title('Amplitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, sr/2)
plt.grid(True)
plt.show()

#A2
time_domain_signal = np.fft.ifft(a)
plt.figure(figsize=(10, 4))
plt.plot(y, label='Original Signal', color='blue')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Generated Time Domain Signal')
plt.legend()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(time_domain_signal, label='Generated Time Domain Signal', color='red', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Original Signal')
plt.legend()
plt.show()
#They're the same, so successful reconstruction done.

#A3 Segmented the word "AI"
start_time = 0
end_time = 1
start_index = int(start_time * sr)
end_index = int(end_time * sr)
word_segment = y[start_index:end_index]
ipd.Audio(word_segment, rate=sr)

word_spectrum = np.fft.fft(word_segment)
freq_word = np.fft.fftfreq(len(word_segment), d=1/sr)

full_spectrum = np.fft.fft(y)
freq_full = np.fft.fftfreq(len(y), d=1/sr)
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(freq_full, np.abs(full_spectrum), label='Full Signal Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Spectrum of Full Signal')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(freq_word, np.abs(word_spectrum), label='Word Spectrum', color='orange')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Spectrum of Word Segment')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

#A4
window_length_ms = 20  
sr2 = 22500
window_length_samples = int(window_length_ms * sr2 / 1000)
speech_window = y[:window_length_samples]

fft_result = np.fft.fft(speech_window)
fft_magnitude = np.abs(fft_result)
fft_freq = np.fft.fftfreq(window_length_samples, 1/sr2)

plt.figure(figsize=(10, 6))
plt.plot(fft_freq, fft_magnitude, color='blue')
plt.title('Spectral Analysis of Speech Signal in 20 ms Window')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()

#A5
window_length_ms = 20  
window_length_samples = int(window_length_ms * sr / 1000)
hop_length = int(window_length_samples / 2)
D = librosa.stft(y, n_fft=window_length_samples, hop_length=hop_length)
spectrogram = np.abs(D)

plt.figure(figsize=(10, 6))
librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram of Speech Signal')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.show()

#A6
from scipy import signal
frequencies, times, spectrogram = signal.spectrogram(y, fs=sr, window='hann', nperseg=window_length_samples, noverlap=hop_length)
plt.figure(figsize=(10, 6))
plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram), shading='gouraud')
plt.title('Spectrogram of Speech Signal(Scipy)')
plt.xlabel('Time(s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Magnitude (dB)')
plt.show()