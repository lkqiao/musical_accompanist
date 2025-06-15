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

# for preprocessing
pedal_reduce = True
pedal_strength = 0.5
bandpass_high = 4000
bandpass_low = 20

# for onset envelope
lowpass_cutoff = 10000

# for tempogram and tempo bins
tempo_min = 35
tempo_max = 200
alpha = 0.075
plot_tempogram = True

# for tempo estimation
peak_threshold = 0.2
window_size = 5
# ----------------------------------------------------------------------------------------------

# preprocess signal
original_signal, signal = damped_bandpass(signal, sr, bandpass_low, bandpass_high, 
                                          pedal_reduce=pedal_reduce, name=name, pedal_strength=pedal_strength)

# get onset envelope
onset_env = get_onset_env(signal, sr, hop_length, lowpass_cutoff, alpha)

# get tempogram and tempo bins
tempogram, tempo_bins, fmin, fmax = get_tempogram_tempo_bins(onset_env, sr, hop_length, win_length, tempo_min=tempo_min, tempo_max=tempo_max, 
                                                 plot_tempogram=plot_tempogram, name=name, time_res=time_res)

# estimate tempo from tempogram
estim_tempos, tempo_t = extract_tempogram_tempos(tempogram, tempo_bins, fmin, fmax, time_res, 
                                                 peak_threshold=peak_threshold, window_size=window_size)
estim_tempos = process_tempos(estim_tempos)

# plot estimated tempos and original signal
plot_estim_tempos(signal, sr, estim_tempos, tempo_t, fmin, fmax, name)

# align click track with first note
duration = len(signal)/sr
click_track = generate_click_track_from_estimates(tempo_t, estim_tempos, sr, duration)
click_signal = synthesize_click_signal(signal, sr, click_track, 
									   pedal_reduce=pedal_reduce, original_signal=original_signal, name=name)