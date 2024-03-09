import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav


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


time_domain_signal = np.fft.ifft(a)

plt.figure(figsize=(10, 4))
plt.plot(time_domain_signal, label='Generated Time Domain Signal', color='red', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Original Signal')
plt.legend()
plt.show()

#A2
D = librosa.stft(y)
magnitude = np.abs(D)
phase = np.angle(D)

# low-pass filtering
rect_window_lowpass = np.zeros_like(magnitude)
rect_window_lowpass[:100, :] = 1
magnitude_lowpass = magnitude * rect_window_lowpass
y_lowpass = librosa.istft(magnitude_lowpass * np.exp(1j * phase))
wav.write('lowpass_filtered_audio.wav', sr, y_lowpass)


# band-pass filtering
rect_window_bandpass = np.zeros_like(magnitude)
rect_window_bandpass[100:300, :] = 1
magnitude_bandpass = magnitude * rect_window_bandpass
y_bandpass = librosa.istft(magnitude_bandpass * np.exp(1j * phase))
wav.write('bandpass_filtered_audio.wav', sr, y_bandpass)

# high-pass filtering
rect_window_highpass = np.zeros_like(magnitude)
rect_window_highpass[300:, :] = 1
magnitude_highpass = magnitude * rect_window_highpass
y_highpass = librosa.istft(magnitude_highpass * np.exp(1j * phase))
wav.write('highpass_filtered_audio.wav', sr, y_highpass)




#A3


#Cosine filters
D = librosa.stft(y)
magnitude = np.abs(D)
phase = np.angle(D)

#low pass
cos_window_lowpass = np.hanning(magnitude.shape[0])[:, None]
magnitude_lowpass = magnitude * cos_window_lowpass
y_lowpass = librosa.istft(magnitude_lowpass * np.exp(1j * phase))
wav.write('lowpass_filtered_audio_cosine.wav', sr, y_lowpass)

#band pass
cos_window_bandpass = np.hanning(magnitude.shape[0])[:, None]
cos_window_bandpass[100:300, :] = 1
magnitude_bandpass = magnitude * cos_window_bandpass
y_bandpass = librosa.istft(magnitude_bandpass * np.exp(1j * phase))
wav.write('bandpass_filtered_audio_cosine.wav', sr, y_bandpass)

#high pass
cos_window_highpass = np.hanning(magnitude.shape[0])[:, None]
cos_window_highpass[:100, :] = 1
magnitude_highpass = magnitude * cos_window_highpass
y_highpass = librosa.istft(magnitude_highpass * np.exp(1j * phase))
wav.write('highpass_filtered_audio_cosine.wav', sr, y_highpass)

#gaussian filters

D = librosa.stft(y)
magnitude = np.abs(D)
phase = np.angle(D)
sigma = 10
center = magnitude.shape[0] // 2

#low pass
gaussian_window_lowpass = signal.gaussian(magnitude.shape[0], std=sigma)
gaussian_window_lowpass /= np.sum(gaussian_window_lowpass)
magnitude_lowpass = magnitude * gaussian_window_lowpass[:, None]
y_lowpass = librosa.istft(magnitude_lowpass * np.exp(1j * phase))
wav.write('lowpass_filtered_audio_gaussian.wav', sr, y_lowpass)

# band pass
gaussian_window_bandpass = signal.gaussian(magnitude.shape[0], std=sigma)
gaussian_window_bandpass /= np.sum(gaussian_window_bandpass)
gaussian_window_bandpass[100:300] = 1
magnitude_bandpass = magnitude * gaussian_window_bandpass[:, None]
y_bandpass = librosa.istft(magnitude_bandpass * np.exp(1j * phase))
wav.write('bandpass_filtered_audio_gaussian.wav', sr, y_bandpass)

#high pass
gaussian_window_highpass = signal.gaussian(magnitude.shape[0], std=sigma)
gaussian_window_highpass /= np.sum(gaussian_window_highpass)
gaussian_window_highpass[:100] = 1
magnitude_highpass = magnitude * gaussian_window_highpass[:, None]
y_highpass = librosa.istft(magnitude_highpass * np.exp(1j * phase))
wav.write('highpass_filtered_audio_gaussian.wav', sr, y_highpass)
