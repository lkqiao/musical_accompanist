import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt, lfilter, hilbert
import os
from math import ceil

# bandpass filter
def bandpass_filter(signal, sr, lowcut=60, highcut=4000, order=5):
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs=sr, output='sos')
    filtered_signal = sosfilt(sos, signal)
    return filtered_signal

# lowpass filter
def lowpass_filter(signal, sr, cutoff=4000, order=5):
    sos = butter(order, cutoff, btype='lowpass', fs=sr, output='sos')
    filtered_signal = sosfilt(sos, signal)
    return filtered_signal

# highpass filter
def highpass_filter(signal, sr, cutoff=60, order=5):
    sos = butter(order, cutoff, btype='highpass', fs=sr, output='sos')
    filtered_signal = sosfilt(sos, signal)
    return filtered_signal

def apply_fir_filter(signal, window):
    window = np.array(window)
    window /= np.sum(window)
    filtered = np.convolve(signal, window, mode='same')
    return filtered

def reduce_pedal(y, sr, output_path, strength=0.5, write_file=True):
    analytic_signal = hilbert(y) # envelope detection using hilbert transform
    envelope = np.abs(analytic_signal)

    # create suppression mask
    env_norm = envelope/np.max(envelope)
    inv_env = 1.0-env_norm
    suppression = 1.0-strength*inv_env
    suppressed_y = y*suppression

    output_y = highpass_filter(suppressed_y, sr, cutoff=80, order=4) # highpass to reduce resonance
    
    if write_file: sf.write(output_path, output_y, sr)
    return y, output_y

# extract tempos from tempogram
def extract_tempogram_tempos(tempogram, tempo_bins, fmin, fmax, peak_threshold=0.3, window_size=5):
    half_window = window_size//2

    # find indices of fmin and fmax in tempo_bins
    fmin_idx = np.where(tempo_bins>=fmin)[0][-1]
    fmax_idx = np.where(tempo_bins<=fmax)[0][0]
    print(f'Using tempo bins in index range [{fmax_idx}, {fmin_idx}]')
    print(f'Using tempo bins in BPM range [{tempo_bins[fmin_idx]:.2f} BPM, {tempo_bins[fmax_idx]:.2f} BPM]')

    estim_tempos = []
    for i in range(tempogram.shape[1]):
        col = np.abs(tempogram[fmax_idx:fmin_idx, i])
        peak_val = np.max(col)
        peak_idx = np.argmax(col) + fmax_idx

        if peak_val > peak_threshold:
            # window around the peak
            start = max(peak_idx-half_window, fmax_idx)
            end = min(peak_idx+half_window+1, fmin_idx)
            window_idxs = np.arange(start, end)
            
            weights = np.abs(tempogram[window_idxs, i])
            if np.sum(weights) > 0: weighted_tempo = np.sum(tempo_bins[window_idxs]*weights)/np.sum(weights)
            else: weighted_tempo = tempo_bins[peak_idx]

            estim_tempos.append(float(weighted_tempo))
        elif i == 0: 
            estim_tempos.append(0)
        else: 
            estim_tempos.append(estim_tempos[i-1])
    return np.array(estim_tempos)

def is_outlier(val, median, mad, threshold):
    return np.abs(val-median) > threshold*mad

def process_tempos(data, threshold=3.5, max_iter=1000):
    cleaned = data.copy()
    for _ in range(max_iter):
        median = np.median(cleaned)
        mad = np.median(np.abs(cleaned-median))

        if mad < 1e-6: break

        outlier_indices = [i for i, val in enumerate(cleaned) if is_outlier(val, median, mad, threshold)]
        if not outlier_indices: break

        for i in outlier_indices:
            if cleaned[i] > median: cleaned[i] /= 2
            elif cleaned[i] < median: cleaned[i] *= 2

    return cleaned

# detect start
def detect_start(signal):
    threshold = 0.01 * np.max(np.abs(signal))
    start_idx = 0
    for i, val in enumerate(signal):
        if np.abs(val) > threshold:
            start_idx = i
            break
    return start_idx

# zero pad signal to start at signal
def zero_pad_signal(signal, start_frame):
    leading_zeros = np.argmax(signal != 0) if np.any(signal != 0) else len(signal)

    if leading_zeros < start_frame:
        pad_amount = start_frame - leading_zeros
        return np.concatenate((np.zeros(pad_amount), signal))
    elif leading_zeros > start_frame:
        return signal[(leading_zeros - start_frame):]
    else:
        return signal

# generate click track to verify estimated tempos
def generate_click_track_from_estimates(tempo_t, tempos, sr, duration, click_dur=50, click_freq=1000):
    click_track = np.zeros(int(sr*duration))
    click_length = int(click_dur/1000*sr)

    change_idxs = np.where(np.floor(np.abs(np.diff(tempos)))!=0)[0]
    change_idxs = np.concatenate(([0], change_idxs+1, [len(tempos)])) # always include the first index

    # start at the first time
    current_time = tempo_t[0] if len(tempo_t) > 0 else 0.0

    for seg_idx in range(len(change_idxs)-1):
        seg_start = change_idxs[seg_idx]
        seg_end = change_idxs[seg_idx+1]
        tempo = tempos[seg_start]
        if tempo <= 0:
            # skip this segment, but advance current_time to the start of next segment
            if seg_end < len(tempo_t):
                current_time = tempo_t[seg_end]
            continue
        t_start = tempo_t[seg_start]
        t_end = tempo_t[seg_end-1] if seg_end < len(tempo_t) else duration
        T = 60.0/tempo
        current_time = max(current_time, t_start) # always start placing clicks at the start of the segment
        while current_time < t_end:
            idx = int(current_time*sr)
            if idx + click_length <= len(click_track):
                click_track[idx:idx+click_length] += 0.5
            elif idx < len(click_track):
                click_track[idx:] += 0.5
            current_time += T # at the end of the segment, current_time is carried over to the next segment

    # modulate click track with sine wave
    sin_wave = np.sin(2*np.pi*click_freq*np.arange(len(click_track))/sr)
    click_track *= sin_wave
    return click_track