import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import os

filename = 'minecraft.mp3'
signal, sr = librosa.load(f'audio_in/{filename}')  

hop_length = 512
win_dur = 30.0 
win_length = int((win_dur*sr)/hop_length)

onset_env = librosa.onset.onset_strength(y=signal, sr=sr, hop_length=hop_length)
if np.all(onset_env == 0): raise ValueError("Onset envelope is all zeros. Try a different audio file.")

tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr,
                                      hop_length=hop_length, win_length=win_length)
tempogram = tempogram[1:,:]
tempos = librosa.tempo_frequencies(tempogram.shape[0], sr=sr, hop_length=hop_length)
times = librosa.frames_to_time(np.arange(tempogram.shape[1]), sr=sr, hop_length=hop_length)

min_bpm, max_bpm = 30, 200

mask = (tempos >= min_bpm) & (tempos <= max_bpm)
tempogram = tempogram[mask,:]
tempos = tempos[mask] 

plt.figure(figsize=(10, 4))
librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length,
                         x_axis='time', y_axis='tempo', y_coords=tempos)
plt.ylim(min(tempos), max(tempos))
plt.title(f'Tempogram (win_dur = {win_dur} s) for {filename}')
plt.colorbar(label='Autocorrelation')
plt.tight_layout()
plt.show()

max_indices = np.argmax(tempogram, axis=0)
dominant_tempos = tempos[max_indices]

plt.figure(figsize=(10, 4))
plt.plot(times, dominant_tempos, label='Dominant Tempo (BPM)')
plt.xlabel('Time (s)')
plt.ylabel('Tempo (BPM)')
plt.title('Tempo (BPM) vs Time (s)')
plt.show()

frame_times = librosa.frames_to_time(np.arange(len(dominant_tempos)), sr=sr, hop_length=hop_length)

beat_periods = 60.0/dominant_tempos 

beat_times = []
for t0, period in zip(frame_times, beat_periods):
    n_beats = int((hop_length/sr))
    beat_times.append(t0)
beat_times = np.unique(np.array(beat_times))

clicks = librosa.clicks(times=beat_times, sr=sr, click_duration=0.03, length=len(signal))
click_overlay = signal + clicks

# Save result
sf.write(f'audio_out/{os.path.splitext(filename)[0]}_clicks.mp3', click_overlay, sr)
