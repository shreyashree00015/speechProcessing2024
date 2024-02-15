import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib as plt

filename = "SP_shreya_audioFile.wav"
#A1
#original
y, sr = librosa.load(filename)
#trimming
yt,index = librosa.effects.trim(y, top_db=20, hop_length=512, frame_length = 2048)
print("Non silent regions",index[0],index[1])
librosa.display.waveshow(yt)
ipd.Audio(yt, rate=sr)

print("The duration of the original audio is",librosa.get_duration(y))
print("The duration of the trimmed audio is", librosa.get_duration(yt))

#A2

top_db = [30,60,90]
# Detect the splites (silent intervals)
for i in top_db:
    print(i)
    intervals = librosa.effects.split(y,top_db=i)
    #splitting
    segments = [y[start:end] for start, end in intervals]
    # Playing each segment and display its waveform
    for i, segment in enumerate(segments):
        print(f"Playing segment {i+1}")
        ipd.display(ipd.Audio(segment, rate=sr))
        librosa.display.waveshow(segment, sr=sr)
        plt.pyplot.show()
