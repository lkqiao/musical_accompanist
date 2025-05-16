import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import os

import numpy as np

def compressor (audio, threshold_db=-20.0, ratio=4.0, makeup_gain_db=0.0, limit_output=True):
    eps = 1e-10 
    magnitude = np.abs(audio)
    audio_db = 20.0 * np.log10(magnitude + eps)

    over_threshold = audio_db > threshold_db
    compressed_db = np.copy(audio_db)
    compressed_db[over_threshold] = threshold_db + (audio_db[over_threshold] - threshold_db) / ratio

    compressed_db += makeup_gain_db

    compressed_mag = 10.0 ** (compressed_db / 20.0)
    compressed_audio = np.sign(audio) * compressed_mag

    if limit_output:
        compressed_audio = np.clip(compressed_audio, -1.0, 1.0)

    return compressed_audio

# parameters
filename = 'minecraft.mp3'
signal, sr = librosa.load(f'audio_in/{filename}', sr=None)  
signal = compressor(signal, ratio=1.25)
print('Sample rate:', sr)
print('Signal length:', len(signal))

hop_length = 2048
win_dur = 45.0 
win_length = int((win_dur*sr)/hop_length)

time_res = hop_length/sr
print('Time resolution:', time_res)

# maybe try wavelet transform instead of stft for onset detection (find 
# power of frequency spectrum of each frame and compute spectral flux) 
# what if each basis function was a wavelet + sin wave?
onset_env = librosa.onset.onset_strength(y=signal, sr=sr, hop_length=hop_length)
if np.all(onset_env == 0): raise ValueError("Onset envelope is all zeros. Try a different audio file.")

# does autocorrelation of the onset envelope to get the tempogram
tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr,
                                      hop_length=hop_length, win_length=win_length)

# get rid of first row, which is the zero lag
tempogram = tempogram[1:,:]

# get the tempos and times from tempogram
tempos = librosa.tempo_frequencies(tempogram.shape[0], sr=sr, hop_length=hop_length)
times = librosa.frames_to_time(np.arange(tempogram.shape[1]), sr=sr, hop_length=hop_length)

# limit tempogram to a range of tempos
min_bpm, max_bpm = 30, 200
mask = (tempos >= min_bpm) & (tempos <= max_bpm)
tempogram = tempogram[mask,:]
tempos = tempos[mask] 

# plot the tempogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length,
                         x_axis='time', y_axis='tempo', y_coords=tempos)
plt.ylim(min(tempos), max(tempos))
plt.title(f'Tempogram (win_dur = {win_dur} s) for {filename}')
plt.colorbar(label='Autocorrelation')
plt.tight_layout()
plt.show()

# find the dominant tempo at each time frame
max_indices = np.argmax(tempogram, axis=0)
dominant_tempos = tempos[max_indices]
print('Dominant tempos:', dominant_tempos)

# plot the dominant tempo vs time
plt.figure(figsize=(10, 4))
plt.plot(times, dominant_tempos, label='Dominant Tempo (BPM)')
plt.xlabel('Time (s)')
plt.ylabel('Tempo (BPM)')
plt.title('Tempo (BPM) vs Time (s)')
plt.show()

# add clicks to the original audio to check
click_times = []
for i, bpm in enumerate(dominant_tempos):
    t_start = i * hop_length / sr
    period = 60.0 / bpm
    num_clicks = int((hop_length / sr) / period) + 1
    for n in range(num_clicks):
        click_time = t_start + n * period
        if click_time < len(signal) / sr:
            click_times.append(click_time)

clicks = librosa.clicks(times=click_times, sr=sr, length=len(signal))
click_overlay = signal + 0.01*clicks

sf.write(f'audio_out/{os.path.splitext(filename)[0]}_clicks.mp3', click_overlay, sr)
