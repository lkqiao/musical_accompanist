import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt, lfilter, hilbert
import os
from math import ceil

from rec_tempo_detect_helpers import *

#  choose audio file
name = 'chopin_berceuse'
filename = f'{name}.mp3'

# load audio file
audio_path = f'audio_in/{filename}'
if not os.path.exists(audio_path):
	raise FileNotFoundError(f'Audio file not found: {audio_path}')

signal: np.ndarray
signal, sr = librosa.load(audio_path, sr=None)  
signal -= np.mean(signal) # remove DC offset
t = np.arange(len(signal))/sr # get time vector

print(f'Sample rate: {sr}')
print(f'Signal length: {len(signal)}')
print(f'Signal duration: {len(signal)/sr:.3f} s')

# ----------------------------------------------------------------------------------------------
# define parameters
hop_length = 256 # distance between frames
win_dur = 2 # window duration in seconds (use 4 for percussive pieces, 2 for classical pieces)

time_res = hop_length/sr # temporal resolution
freq_res = 1/win_dur # frequency resolution
win_length = int((win_dur*sr)/hop_length) # window length in frames
print(f'Temporal resolution: {time_res:.3f} s')
print(f'Frequency resolution: {freq_res:.3f} Hz/bin')
print(f'Window length: {win_length} frames')

pedal_reduce = True
pedal_strength = 0.5

bandpass_high = 4000
bandpass_low = 20
lowpass_cutoff = 10000

tempo_min = 35
tempo_max = 200
alpha = 0.075
plot_tempogram = True

peak_threshold = 0.2
# ----------------------------------------------------------------------------------------------

# reduce pedal
original_signal = None
if pedal_reduce:
    dampened_path = f'audio_out/{name}_dampened.mp3'
    original_signal, signal = reduce_pedal(signal, sr, dampened_path, strength=pedal_strength)
    signal /= np.max(np.abs(signal))

# bandpass filter original signal
signal = bandpass_filter(signal, sr, lowcut=bandpass_low, highcut=bandpass_high)
signal /= np.max(np.abs(signal)) # normalize

# find onset envelope
onset_env = librosa.onset.onset_strength(y=signal, sr=sr, hop_length=hop_length)
onset_env -= np.mean(onset_env)
if np.all(onset_env == 0): 
    raise ValueError('Onset envelope is all zeros. Try a different audio file.')

# filter onset envelope
onset_env = lowpass_filter(onset_env, sr, cutoff=lowpass_cutoff, order=8)

# keep values avove threshold
threshold = alpha*np.max(onset_env)
threshold_idx = np.where(onset_env<threshold)[0]
onset_env[threshold_idx] = 0

# compute tempogram, skipping first bin
tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length, win_length=win_length)[1:,:]
tempo_bins = librosa.tempo_frequencies(tempogram.shape[0], sr=sr, hop_length=hop_length)

# for limiting tempo of tempogram
fmin = max(tempo_min, tempo_bins[-1])
fmax = min(tempo_max, tempo_bins[0])

print(f'Max tempo: {fmax:.2f} BPM')
print(f'Min tempo: {fmin:.2f} BPM')
print(f'Number of tempo bins: {len(tempo_bins)}')

# plot tempogram
if plot_tempogram:
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length, x_axis='time', 
                            y_axis='tempo', cmap='inferno')
    plt.ylim(fmin, fmax) 
    plt.colorbar(label='Amplitude')
    plt.title(f'{name} Tempogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Tempo (BPM)')

    yticks = np.arange(ceil(fmin), ceil(fmax), 10)
    _ = plt.yticks(yticks)
    _ = plt.xticks(np.linspace(0, tempogram.shape[1]*time_res, 20))
    plt.show()

estim_tempos = extract_tempogram_tempos(tempogram, tempo_bins, fmin, fmax, peak_threshold=peak_threshold)
print(f'Estimated tempos: {estim_tempos}')
tempo_t = np.arange(len(estim_tempos))*time_res
estim_tempos = process_tempos(estim_tempos)

# plot estimated tempos and original signal
fig, ax1 = plt.subplots(figsize=(12, 4))
librosa.display.waveshow(signal, sr=sr, ax=ax1, label='Signal')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude (Signal)', color='b')
ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()
ax2.plot(tempo_t, estim_tempos, color='r', label='Estimated Tempo')
ax2.set_ylabel('Estimated Tempo (BPM)', color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax2.set_yticks(np.arange(int(fmin), int(fmax), 10))
plt.title(f'{name} Estimated Tempos and Signal')
_ = fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.85))
plt.show()

# align click track with first note
start_frame = detect_start(signal)
duration = len(signal)/sr
click_track = generate_click_track_from_estimates(tempo_t, estim_tempos, sr, duration)
click_track_aligned = zero_pad_signal(click_track, start_frame)

combined = signal + 0.5 * click_track_aligned[:len(signal)]
if pedal_reduce:
    combined = original_signal + 0.5 * click_track_aligned[:len(signal)]

# save click track to file
click_track_path = f'audio_out/click_{name}.mp3'
sf.write(click_track_path, combined, sr)
print(f'Click track saved to: {click_track_path}')

# plot click track
plt.figure(figsize=(12, 4))
plt.plot(np.arange(len(combined))/sr, combined)
plt.title(f'Click Track for {name} (Aligned)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()