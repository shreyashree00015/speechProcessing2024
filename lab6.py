#importing required libraries
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

#functions to reuse
def plotting(freq,spectrum,sampling_rate):
    plt.figure(figsize=(10, 5))
    plt.plot(freq, np.abs(spectrum))
    plt.title('Amplitude Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim(0, sampling_rate/2)
    plt.grid(True)
    plt.show()

def spectrogram_calculate_and_plot(audioSignal, sampling_rate):
    window_length_ms = 20
    window_length_samples = int(window_length_ms * sampling_rate / 1000)
    hop_length = int(window_length_samples / 2)
    D = librosa.stft(audioSignal, n_fft=window_length_samples, hop_length=hop_length)
    spectrogram = np.abs(D)
    #plot
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=sampling_rate, hop_length=hop_length, x_axis='time', y_axis='linear')
    plt.title('Spectrogram of Speech Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()


#calling the file
filename = "SP_shreya_audioFile.wav"
audioSignal, sampling_rate = librosa.load(filename)

#segment: A in AI
#the word A was found to be between 0.5s and 0.85s
segment1 = audioSignal[int(0.5*sampling_rate):int(0.85*sampling_rate)]
ipd.Audio(segment1, rate=sampling_rate)

#performing fft on the vowel
spectrumSegment1 = np.fft.fft(segment1)
#extraing the frequency components
freqComponents = np.fft.fftfreq(len(spectrumSegment1), d=1/sampling_rate)
plotting(freqComponents,spectrumSegment1,sampling_rate)



#segment: 's' in speech
#the word 's' in speech was found to be between 1.25s and 1.3s
segment2 = audioSignal[int(1.25*sampling_rate):int(1.3*sampling_rate)]
ipd.Audio(segment2, rate=sampling_rate)
#performing fft on the consonant
spectrumSegment2 = np.fft.fft(segment2)
#extraing the frequency components 
freqComponents = np.fft.fftfreq(len(spectrumSegment2), d=1/sampling_rate)
plotting(freqComponents,spectrumSegment2,sampling_rate)


#performing the same for slices of silence and non voiced portion
segment3 = audioSignal[int(0.1*sampling_rate):int(0.4*sampling_rate)]
ipd.Audio(segment3, rate=sampling_rate)
#performing fft on the non voiced portion
spectrumSegment3 = np.fft.fft(segment3)
#extraing the frequency components 
freqComponents = np.fft.fftfreq(len(spectrumSegment3), d=1/sampling_rate)
plotting(freqComponents,spectrumSegment3,sampling_rate)


#spectrogram generation
spectrogram_calculate_and_plot(audioSignal, sampling_rate)
