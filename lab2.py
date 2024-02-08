import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

filename = "SP_shreya_audioFile.wav"
speech_signal, sr = sf.read(filename)
time = np.arange(len(speech_signal)) / sr

#A1
firstDerivative = np.diff(speech_signal) / np.diff(time)
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, speech_signal, color='b')
plt.title('Original Speech Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(time[:-1], firstDerivative, color='r')
plt.title('First Derivative of Speech Signal')
plt.xlabel('Time (s)')
plt.ylabel('First Derivative')
sf.write('first_derivative_signal.wav', firstDerivative, sr)

#A2
speech_derivative, sr = sf.read('first_derivative_signal.wav')
zero_crossing_indices = []
for i in range(len(speech_derivative) - 1):
    difference = speech_derivative[i + 1] - speech_derivative[i]
    if difference != 0:
        zero_crossing_indices.append(i)

zero_crossing_lengths = np.diff(zero_crossing_indices)
speech_threshold = 0.1
silence_threshold = 0.01  

speech_regions = zero_crossing_lengths[zero_crossing_lengths > speech_threshold * sr]
silence_regions = zero_crossing_lengths[zero_crossing_lengths <= silence_threshold * sr]

avg_length_speech = np.mean(speech_regions)
avg_length_silence = np.mean(silence_regions)

print("Length between two consecutive zero crossings in speech regions:", avg_length_speech / sr, "seconds")
print("Length between two consecutive zero crossings in silence regions:", avg_length_silence / sr, "seconds")

#A3
filename_teammate1 = "Priyanka_fries.wav"
filename_teammate2 = "shreya_fries.wav"

y1, sr1 = librosa.load(filename_teammate1)
librosa.display.waveshow(y1)
ipd.Audio(y1, rate=sr1)

y2, sr2 = librosa.load(filename_teammate2)
librosa.display.waveshow(y2)
ipd.Audio(y2, rate=sr2)

#teammate 2 spoke took lesser time to speak the same 5 words(Shreya priyanka peri peri fries).

#A4

filename_1 = "i am stupid. ~shreya.wav"
filename_2 = "i am stupid ~ shreya.wav"

y1, sr1 = librosa.load(filename_1)
librosa.display.waveshow(y1)
ipd.Audio(y1, rate=sr1)

y2, sr2 = librosa.load(filename_2)
librosa.display.waveshow(y2)
ipd.Audio(y2, rate=sr2)