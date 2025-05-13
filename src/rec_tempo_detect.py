import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

filename = 'minecraft.mp3'
signal, sr = librosa.load(f'audio/{filename}')  

hop_length = 512
win_dur = 30.0 
win_length = int((win_dur*sr)/hop_length)

onset_env = librosa.onset.onset_strength(y=signal, sr=sr, hop_length=hop_length)
tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr,
                                      hop_length=hop_length, win_length=win_length)

plt.figure(figsize=(10, 4))
librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length,
                         x_axis='time', y_axis='tempo')
plt.title(f'Tempogram (win_dur = {win_dur} s) for {filename}')
plt.colorbar(label='Autocorrelation')
plt.tight_layout()
plt.show()
