import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt, hilbert
from math import ceil

def bandpass_filter(signal, sr, lowcut=60, highcut=4000, order=5):
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs=sr, output='sos') # bandpass filter
    filtered_signal = sosfilt(sos, signal)
    return filtered_signal

def lowpass_filter(signal, sr, cutoff=4000, order=5):
    sos = butter(order, cutoff, btype='lowpass', fs=sr, output='sos') # lowpass filter
    filtered_signal = sosfilt(sos, signal)
    return filtered_signal

def highpass_filter(signal, sr, cutoff=60, order=5):
    sos = butter(order, cutoff, btype='highpass', fs=sr, output='sos') # highpass filter
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

def damped_bandpass(signal, sr, bandpass_low, bandpass_high, pedal_reduce=False, name=None, pedal_strength=0.5):
    print('\n----------------- Preprocessing Signal -----------------\n')
    # reduce pedal
    original_signal = None
    if pedal_reduce:
        dampened_path = f'audio_out/partial_renders/{name}_dampened.mp3'
        original_signal, signal = reduce_pedal(signal, sr, dampened_path, strength=pedal_strength)
        signal /= np.max(np.abs(signal))

    # bandpass filter original signal
    signal = bandpass_filter(signal, sr, lowcut=bandpass_low, highcut=bandpass_high)
    signal /= np.max(np.abs(signal)) # normalize
    return original_signal, signal

def get_onset_env(signal, sr, hop_length, lowpass_cutoff, alpha):
    print('\n----------------- Getting Onset Envelope -----------------\n')
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
    return onset_env

def get_tempogram_tempo_bins(onset_env, sr, hop_length, win_length, tempo_min=35, tempo_max=200, plot_tempogram=False, name=None, time_res=0):
    print('\n----------------- Computing Tempogram and Tempo Bins -----------------\n')
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
        plt.figure(figsize=(12, 7))
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
    
    return tempogram, tempo_bins, fmin, fmax

# extract tempos from tempogram
def extract_tempogram_tempos(tempogram, tempo_bins, fmin, fmax, time_res, peak_threshold=0.3, window_size=5):
    print('\n----------------- Extracting Tempos -----------------\n')
    half_window = window_size//2

    # find indices of fmin and fmax in tempo_bins
    fmin_idx = np.where(tempo_bins>=fmin)[0][-1]
    fmax_idx = np.where(tempo_bins<=fmax)[0][0]
    print(f'Using tempo bins in index range [{fmax_idx}, {fmin_idx}]')
    print(f'Using tempo bins in BPM range [{tempo_bins[fmin_idx]:.2f} BPM, {tempo_bins[fmax_idx]:.2f} BPM]')

    estim_tempos = []
    for i in range(tempogram.shape[1]):
        print(f'Extracting tempos from tempogram: {(i+1)/tempogram.shape[1]*100:.1f}%', end='\r')
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

    tempo_t = np.arange(len(estim_tempos))*time_res
    return np.array(estim_tempos), tempo_t

def is_outlier(val, median, mad, threshold):
    return np.abs(val-median) > threshold*mad

def process_tempos(data, threshold=3.5, max_iter=1000):
    print('\n----------------- Processing tempos -----------------\n')

    cleaned = data.copy()
    curr_max = max_iter/10
    for i in range(max_iter):
        if i >= curr_max: curr_max *= 10
        else: print(f'Processing tempos: {(i+1)/curr_max*100:.1f}%', end='\r')

        median = np.median(cleaned)
        mad = np.median(np.abs(cleaned-median))

        if mad < 1e-6: break

        outlier_indices = [i for i, val in enumerate(cleaned) if is_outlier(val, median, mad, threshold)]
        if not outlier_indices: break

        for i in outlier_indices:
            if cleaned[i] > median: cleaned[i] /= 2
            elif cleaned[i] < median: cleaned[i] *= 2

    return cleaned

def plot_estim_tempos(signal, sr, estim_tempos, tempo_t, fmin, fmax, name):
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

def synthesize_click_signal(signal, sr, click_track, pedal_reduce=False, original_signal=None, name=None):
    start_frame = detect_start(signal)
    click_track_aligned = zero_pad_signal(click_track, start_frame)

    combined = None
    if pedal_reduce: 
        min_len = min(len(original_signal), len(click_track_aligned))
        combined = original_signal[:min_len] + 0.5 * click_track_aligned[:min_len]
    else:
        min_len = min(len(signal), len(click_track_aligned))
        combined = signal[:min_len] + 0.5 * click_track_aligned[:min_len]

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

    return combined