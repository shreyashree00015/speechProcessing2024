import librosa
import librosa.display
import IPython.display as ipd
import numpy as np

filename = "D://Uni//6th Sem//SP//SP_shreya_audioFile.wav"

y, sr = librosa.load(filename)
#whole audio
librosa.display.waveshow(y)
ipd.Audio(y, rate=sr)

#segment: ai
segment1 = y[int(0.5*sr):int(1.0*sr)]
ipd.Audio(segment1, rate=sr)

#segment: processing
segment2 = y[int(1.5*sr):int(2.5*sr)]
ipd.Audio(segment2, rate=sr)

#reverse of the whole audio
reversed_audio = np.flip(y)
librosa.display.waveshow(reversed_audio)

#pitch of the whole audio file
pitch_2 = 2.0
pitch1 = librosa.effects.pitch_shift(y, sr, n_steps=pitch_2)
ipd.Audio(pitch1, rate=sr)

#pitch of the whole audio file
pitch_3 = 3.0
pitch2 = librosa.effects.pitch_shift(segment1, sr, n_steps=pitch_3)
ipd.Audio(pitch2, rate=sr)

#reversed audio with pitch
reversed_audio2 = np.flip(pitch1)
ipd.Audio(reversed_audio2, rate=sr)