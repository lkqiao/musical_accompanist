import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt, hilbert
from math import ceil

def bandpass_filter(signal, sr, lowcut=60, highcut=4000, order=5):
    '''
    Applies a bandpass Butterworth filter to the input signal.

    Parameters:
        signal (array-like): The input audio signal to be filtered.
        sr (int): The sampling rate of the audio signal.
        lowcut (float, optional): The lower frequency cutoff for the bandpass filter in Hz. Default is 60.
        highcut (float, optional): The upper frequency cutoff for the bandpass filter in Hz. Default is 4000.
        order (int, optional): The order of the Butterworth filter. Default is 5.

    Returns:
        numpy.ndarray: The filtered audio signal.
    '''
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs=sr, output='sos') 
    filtered_signal = sosfilt(sos, signal)
    return filtered_signal

def lowpass_filter(signal, sr, cutoff=4000, order=5):
    '''
    Applies a low-pass Butterworth filter to the input signal.

    Parameters:
        signal (array-like): The input audio signal to be filtered.
        sr (int): The sampling rate of the audio signal.
        cutoff (float, optional): The cutoff frequency of the low-pass filter in Hz. Defaults to 4000.
        order (int, optional): The order of the Butterworth filter. Defaults to 5.

    Returns:
        numpy.ndarray: The filtered audio signal.
    '''
    sos = butter(order, cutoff, btype='lowpass', fs=sr, output='sos')
    filtered_signal = sosfilt(sos, signal)
    return filtered_signal

def highpass_filter(signal, sr, cutoff=60, order=5):
    '''
    Applies a high-pass Butterworth filter to the input signal.

    Args:
        signal (np.ndarray): The input audio signal to be filtered.
        sr (int): The sampling rate of the audio signal.
        cutoff (float, optional): The cutoff frequency of the high-pass filter in Hz. Defaults to 60.
        order (int, optional): The order of the Butterworth filter. Defaults to 5.

    Returns:
        np.ndarray: The filtered audio signal after applying the high-pass filter.
    '''
    sos = butter(order, cutoff, btype='highpass', fs=sr, output='sos') 
    filtered_signal = sosfilt(sos, signal)
    return filtered_signal

def apply_fir_filter(signal, window):
    '''
    Applies a Finite Impulse Response (FIR) filter to the input signal using the specified window.

    Parameters:
        signal (array-like): The input signal to be filtered.
        window (array-like): The FIR filter coefficients (window). Will be normalized to sum to 1.

    Returns:
        numpy.ndarray: The filtered signal, same length as the input signal.
    '''
    window = np.array(window)
    window /= np.sum(window)
    filtered = np.convolve(signal, window, mode='same')
    return filtered

def reduce_pedal(y, sr, output_path, strength=0.5, write_file=True):
    '''
    Reduces the effect of piano pedal resonance in an audio signal.
    This function applies envelope detection using the Hilbert transform to estimate the pedal resonance,
    then suppresses the resonance by attenuating the signal where the envelope is high. A highpass filter
    is applied to further reduce low-frequency resonance.
    Args:
        y (np.ndarray): Input audio signal.
        sr (int): Sample rate of the audio signal.
        output_path (str): Path to save the processed audio file.
        strength (float, optional): Strength of pedal resonance suppression (0.0 to 1.0). Default is 0.5.
        write_file (bool, optional): Whether to write the processed audio to a file. Default is True.
    Returns:
        tuple: (original audio signal, processed audio signal)
    '''
    analytic_signal = hilbert(y) # envelope detection using hilbert transform
    envelope = np.abs(analytic_signal)

    # create suppression mask
    env_norm = envelope/np.max(envelope)
    inv_env = 1.0-env_norm
    suppression = 1.0-strength*inv_env
    suppressed_y = y*suppression

    output_y = highpass_filter(suppressed_y, sr, cutoff=80, order=4)  # highpass to reduce resonance
    
    if write_file: sf.write(output_path, output_y, sr)
    return y, output_y

def damped_bandpass(signal, sr, bandpass_low, bandpass_high, pedal_reduce=False, name=None, pedal_strength=0.5):
    '''
    Preprocesses an audio signal by optionally reducing pedal effects and applying a bandpass filter.

    Parameters:
        signal (np.ndarray): The input audio signal.
        sr (int): The sampling rate of the audio signal.
        bandpass_low (float): The lower cutoff frequency for the bandpass filter (in Hz).
        bandpass_high (float): The upper cutoff frequency for the bandpass filter (in Hz).
        pedal_reduce (bool, optional): Whether to apply pedal reduction to the signal. Defaults to False.
        name (str, optional): The name used for saving the dampened audio file if pedal reduction is applied.
        pedal_strength (float, optional): The strength of the pedal reduction effect (between 0 and 1). Defaults to 0.5.

    Returns:
        tuple:
            original_signal (np.ndarray or None): The original signal before pedal reduction, or None if pedal reduction is not applied.
            signal (np.ndarray): The processed and normalized signal after bandpass filtering.
    '''
    print('\n----------------- Preprocessing Signal -----------------')
    # reduce pedal
    original_signal = None
    if pedal_reduce:
        dampened_path = f'audio_out/partial_renders/{name}_dampened.mp3'
        original_signal, signal = reduce_pedal(signal, sr, dampened_path, strength=pedal_strength)
        signal /= np.max(np.abs(signal))

    # bandpass filter original signal
    signal = bandpass_filter(signal, sr, lowcut=bandpass_low, highcut=bandpass_high)
    signal /= np.max(np.abs(signal))  # normalize
    return original_signal, signal

def get_onset_env(signal, sr, hop_length, lowpass_cutoff, onset_threshold, use_max=True, mean_max_weight=(0.5,0.5)):
    '''
    Computes the onset envelope of an audio signal using librosa's onset detection.

    Args:
        signal (np.ndarray): The input audio signal.
        sr (int): The sampling rate of the audio signal.
        hop_length (int): Number of samples between successive onset envelope values.
        lowpass_cutoff (float): Cutoff frequency (Hz) for the low-pass filter applied to the onset envelope.
        onset_threshold (float): Threshold (as a fraction of the max) below which onset envelope values are set to zero.
        use_max (bool, optional): If True, combines mean and max onset strength aggregations. Defaults to True.
        mean_max_weight (tuple, optional): Weights for combining mean and max onset strength aggregations. Defaults to (0.5, 0.5).

    Returns:
        np.ndarray: The processed onset envelope.
    '''
    print('\n\n----------------- Getting Onset Envelope -----------------')
    # find onset envelope    
    onset_env = librosa.onset.onset_strength(y=signal, sr=sr, hop_length=hop_length, aggregate=np.mean)
    onset_env /= np.max(onset_env + 1e-9)
    if use_max:
        onset_env_max = librosa.onset.onset_strength(y=signal, sr=sr, hop_length=hop_length, aggregate=np.max)
        onset_env_max /= np.max(onset_env_max + 1e-9)
        onset_env = mean_max_weight[0]*onset_env + mean_max_weight[1]*onset_env_max
    onset_env -= np.mean(onset_env)
    if np.all(onset_env == 0): 
        raise ValueError('Onset envelope is all zeros. Try a different audio file.')

    # filter onset envelope
    onset_env = lowpass_filter(onset_env, sr, cutoff=lowpass_cutoff, order=8)

    # keep values avove threshold
    threshold = onset_threshold*np.max(onset_env)
    threshold_idx = np.where(onset_env<threshold)[0]
    onset_env[threshold_idx] = 0
    return onset_env

def get_tempogram_tempo_bins(onset_env, sr, hop_length, win_length, tempo_min=35, tempo_max=200, plot_tempogram=False, name=None, time_res=0, max_wait_time=15):
    '''
    Compute the tempogram and corresponding tempo bins from an onset envelope.
    This function calculates the tempogram (a time–tempo representation) of an audio signal's onset envelope,
    and returns the tempogram matrix, the tempo bins (in BPM), and the effective tempo range used for plotting.
    Optionally, it can plot the tempogram for visualization.
    Args:
        onset_env (np.ndarray): Onset envelope of the audio signal.
        sr (int): Sampling rate of the audio signal.
        hop_length (int): Number of samples between successive onset envelope values.
        win_length (int): Window length (in frames) for tempogram computation.
        tempo_min (float, optional): Minimum tempo (in BPM) to display/consider. Defaults to 35.
        tempo_max (float, optional): Maximum tempo (in BPM) to display/consider. Defaults to 200.
        plot_tempogram (bool, optional): Whether to plot the tempogram. Defaults to False.
        name (str, optional): Name to display in the plot title. Defaults to None.
        time_res (float, optional): Time resolution per frame (in seconds) for x-axis ticks. Defaults to 0.
        max_wait_time (float, optional): Maximum time (in seconds) to display the plot. Defaults to 15.
    Returns:
        tempogram (np.ndarray): The computed tempogram matrix (tempo x time).
        tempo_bins (np.ndarray): Array of tempo bin centers (in BPM).
        (fmin, fmax) (tuple): Tuple containing the effective minimum and maximum tempo (in BPM) used for plotting.
    Notes:
        - The first tempo bin is skipped in the tempogram computation.
        - The function prints information about the computed tempo range and number of bins.
        - If plot_tempogram is True, the function displays the tempogram plot for `max_wait_time` seconds.
    '''
    print('\n\n----------------- Computing Tempogram and Tempo Bins -----------------')
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
        librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='tempo', cmap='inferno')
        plt.ylim(fmin, fmax) 
        plt.colorbar(label='Amplitude')
        plt.title(f'{name} Tempogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Tempo (BPM)')

        yticks = np.arange(ceil(fmin), ceil(fmax), 10)
        _ = plt.yticks(yticks)
        _ = plt.xticks(np.linspace(0, tempogram.shape[1]*time_res, 20))
        print('Generating tempogram plot...')
        plt.show(block=False)
        plt.pause(max_wait_time)
    
    return tempogram, tempo_bins, (fmin, fmax)

def extract_tempogram_tempos(tempogram, tempo_bins, fmin, fmax, time_res, peak_threshold=0.3, window_size=5):
    '''
    Extracts estimated tempo values over time from a tempogram matrix.
    This function analyzes a tempogram (time-frequency representation of tempo strength)
    and extracts the most prominent tempo at each time frame, using a weighted average
    around the detected peak within a specified tempo range.
    Args:
        tempogram (np.ndarray): 2D array (tempo_bins x time) representing the tempogram.
        tempo_bins (np.ndarray): 1D array of tempo bin values (e.g., BPM) corresponding to tempogram rows.
        fmin (float): Minimum tempo (in BPM) to consider for peak detection.
        fmax (float): Maximum tempo (in BPM) to consider for peak detection.
        time_res (float): Time resolution (in seconds) per tempogram frame.
        peak_threshold (float, optional): Minimum peak value to consider a valid tempo (default: 0.3).
        window_size (int, optional): Number of bins around the peak to use for weighted averaging (default: 5).
    Returns:
        tuple:
            - np.ndarray: Array of estimated tempo values (in BPM) for each time frame.
            - np.ndarray: Array of time values (in seconds) corresponding to each tempo estimate.
    Notes:
        - If no peak above the threshold is found, the previous tempo is repeated (or zero for the first frame).
        - The function prints progress and diagnostic information to the console.
    '''
    print('\n\n----------------- Extracting Tempos -----------------')
    half_window = window_size//2

    # find indices of fmin and fmax in tempo_bins
    fmin_idx = np.where(tempo_bins>=fmin)[0][-1]
    fmax_idx = np.where(tempo_bins<=fmax)[0][0]
    print(f'Using tempo bins in index range [{fmax_idx}, {fmin_idx}]')
    print(f'Using tempo bins in BPM range [{tempo_bins[fmin_idx]:.2f} BPM, {tempo_bins[fmax_idx]:.2f} BPM]')

    estim_tempos = []
    for i in range(tempogram.shape[1]):
        bar_length = 30
        progress = (i+1)/tempogram.shape[1]
        filled_length = int(bar_length*progress)
        bar = '=' * filled_length + '-' * (bar_length-filled_length)
        print(f'Extracting tempos: [{bar}] ', end='\r')
        
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
    '''
    Determine if a value is an outlier based on the median absolute deviation (MAD) method.

    Args:
        val (float): The value to check.
        median (float): The median of the dataset.
        mad (float): The median absolute deviation of the dataset.
        threshold (float): The number of MADs a value must differ from the median to be considered an outlier.

    Returns:
        bool: True if the value is an outlier, False otherwise.
    '''
    return np.abs(val-median) > threshold*mad

def process_tempos(data, threshold=3.5, step_num=1, max_iter=1000):
    '''
    Processes a list or array of tempo values by iteratively detecting and correcting outliers.

    This function uses the median and median absolute deviation (MAD) to identify outliers in the input data.
    Outliers are defined as values that deviate from the median by more than `threshold` times the MAD.
    Detected outliers are adjusted: values above the median are halved, and values below the median are doubled.
    The process repeats for a maximum of `max_iter` iterations or until no more outliers are found.

    Args:
        data (list or np.ndarray): The input tempo values to process.
        threshold (float, optional): The number of MADs a value must differ from the median to be considered an outlier. Default is 3.5.
        step_num (int, optional): The batch or step number for display/logging purposes. Default is 1.
        max_iter (int, optional): The maximum number of iterations to perform. Default is 1000.

    Returns:
        list or np.ndarray: The cleaned tempo values with outliers adjusted.
    '''
    print(f'\n\n----------------- Processing Tempos (Batch {step_num}) -----------------')

    cleaned = data.copy()
    curr_max = max_iter/10
    for i in range(max_iter):
        # progress bar
        if i >= curr_max:
            curr_max *= 10
        else:
            bar_length = 30
            progress = (i+1)/curr_max
            filled_length = int(bar_length*progress)
            bar = '=' * filled_length + '-' * (bar_length-filled_length)
            print(f'Processing tempos: [{bar}] ', end='\r')

        median = np.median(cleaned)
        mad = np.median(np.abs(cleaned-median))

        if mad < 1e-6: break

        outlier_indices = [i for i, val in enumerate(cleaned) if is_outlier(val, median, mad, threshold)]
        if not outlier_indices: break

        for i in outlier_indices:
            if cleaned[i] > median: cleaned[i] /= 2
            elif cleaned[i] < median: cleaned[i] *= 2

    return cleaned

def remove_spikes(data, threshold=2):
    '''
    Removes spikes from a 1D NumPy array by replacing outliers (based on z-score threshold) with interpolated values.

    Parameters:
        data (np.ndarray): Input 1D array of numerical values.
        threshold (float, optional): Z-score threshold to identify spikes. Values with absolute z-score greater than this are considered outliers. Default is 2.

    Returns:
        np.ndarray: Array with spikes removed and replaced by interpolated values.
    '''
    print('\n\n----------------- Postprocessing Tempos -----------------')
    mean = np.mean(data)
    std = np.std(data)

    z_scores = (data-mean)/std
    cleaned_nan = np.copy(data)
    cleaned_nan[np.abs(z_scores)>threshold] = np.nan 

    not_nan = ~np.isnan(cleaned_nan)
    cleaned = np.interp(np.arange(len(data)), np.arange(len(data))[not_nan], cleaned_nan[not_nan])
    return cleaned

def plot_estim_tempos(signal, sr, estim_tempos, tempo_t, fmin, fmax, name, max_wait_time=7.5):
    '''
    Plots the estimated tempo curve alongside the original audio signal waveform.

    Parameters:
        signal (np.ndarray): The original audio signal.
        sr (int): The sampling rate of the audio signal.
        estim_tempos (np.ndarray): Array of estimated tempo values (in BPM) over time.
        tempo_t (np.ndarray): Array of time values (in seconds) corresponding to each tempo estimate.
        fmin (float): Minimum tempo (in BPM) for y-axis ticks.
        fmax (float): Maximum tempo (in BPM) for y-axis ticks.
        name (str): Name to display in the plot title.
        max_wait_time (float, optional): Maximum time (in seconds) to display the plot. Defaults to 7.5.

    Returns:
        None
    '''
    # plot estimated tempos and original signal
    fig, ax1 = plt.subplots(figsize=(12, 4))
    librosa.display.waveshow(signal, sr=sr, ax=ax1, label='Signal')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude (Signal)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    duration = len(signal) / sr
    x_ticks = np.arange(0, duration+10, 10)
    ax1.set_xticks(x_ticks)
    plt.setp(ax1.get_xticklabels(), rotation=60, ha='right')

    ax2 = ax1.twinx()
    ax2.plot(tempo_t, estim_tempos, color='r', label='Estimated Tempo')
    ax2.set_ylabel('Estimated Tempo (BPM)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_yticks(np.arange(int(fmin), int(fmax), 10))
    plt.title(f'{name} Estimated Tempos and Signal')
    _ = fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.85))
    print('Generating estimated tempos plot...')
    plt.show(block=False)
    plt.pause(max_wait_time)

def detect_start(signal):
    '''
    Detects the starting index of a signal where its amplitude first exceeds a threshold.

    The threshold is set to 1% of the maximum absolute value in the signal. The function iterates
    through the signal and returns the index of the first sample whose absolute value exceeds this threshold.

    Args:
        signal (np.ndarray): Input 1D signal array.

    Returns:
        int: Index of the first sample exceeding the threshold.
    '''
    threshold = 0.01 * np.max(np.abs(signal))
    start_idx = 0
    for i, val in enumerate(signal):
        if np.abs(val) > threshold:
            start_idx = i
            break
    return start_idx

def zero_pad_signal(signal, start_frame):
    '''
    Adjusts the leading zeros of a signal array so that the first non-zero element appears at a specified frame index.

    Parameters:
        signal (np.ndarray): The input 1D signal array.
        start_frame (int): The desired index for the first non-zero element.

    Returns:
        np.ndarray: The adjusted signal array, either zero-padded or truncated at the beginning so that the first non-zero element is at start_frame.
    '''
    leading_zeros = np.argmax(signal != 0) if np.any(signal != 0) else len(signal)

    if leading_zeros < start_frame:
        pad_amount = start_frame - leading_zeros
        return np.concatenate((np.zeros(pad_amount), signal))
    elif leading_zeros > start_frame:
        return signal[(leading_zeros - start_frame):]
    else:
        return signal

def generate_click_track_from_estimates(tempo_t, tempos, sr, duration, click_dur=50, click_freq=1000):
    '''
    Generate a click track audio signal based on estimated tempo changes over time.

    This function creates a click track (an array of audio samples) where clicks are placed at intervals
    determined by the provided tempo estimates. The click track can handle segments with varying tempos,
    and each click is modulated by a sine wave at a specified frequency.

    Args:
        tempo_t (np.ndarray): Array of time points (in seconds) corresponding to each tempo estimate.
        tempos (np.ndarray): Array of tempo estimates (in beats per minute) for each segment.
        sr (int): Sample rate of the audio (samples per second).
        duration (float): Total duration of the click track (in seconds).
        click_dur (int, optional): Duration of each click (in milliseconds). Default is 50 ms.
        click_freq (int, optional): Frequency of the click sound (in Hz). Default is 1000 Hz.

    Returns:
        np.ndarray: The generated click track as a 1D NumPy array of audio samples.
    '''
    click_track = np.zeros(int(sr*duration))
    click_length = int(click_dur/1000*sr)

    change_idxs = np.where(np.floor(np.abs(np.diff(tempos)))!=0)[0]
    change_idxs = np.concatenate(([0], change_idxs+1, [len(tempos)]))  # always include the first index

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
        current_time = max(current_time, t_start)  # always start placing clicks at the start of the segment
        while current_time < t_end:
            idx = int(current_time*sr)
            if idx + click_length <= len(click_track):
                click_track[idx:idx+click_length] += 0.5
            elif idx < len(click_track):
                click_track[idx:] += 0.5
            current_time += T  # at the end of the segment, current_time is carried over to the next segment

    # modulate click track with sine wave
    sin_wave = np.sin(2*np.pi*click_freq*np.arange(len(click_track))/sr)
    click_track *= sin_wave
    return click_track

def synthesize_click_signal(signal, sr, click_track, pedal_reduce=False, original_signal=None, name=None, max_wait_time=7.5):
    '''
    Synthesizes and saves a click track signal aligned with the input audio signal.

    This function aligns a click track with the detected start of the input signal, optionally reduces pedal noise,
    combines the signals, saves the result to an audio file, and plots the combined waveform.

    Args:
        signal (np.ndarray): The input audio signal array.
        sr (int): The sample rate of the audio signals.
        click_track (np.ndarray): The click track signal to be aligned and combined.
        pedal_reduce (bool, optional): If True, uses the original signal for pedal noise reduction. Defaults to False.
        original_signal (np.ndarray, optional): The original audio signal, required if pedal_reduce is True.
        name (str, optional): Name identifier for saving the output file and plot title.
        max_wait_time (float, optional): Maximum time in seconds to display the plot. Defaults to 7.5.

    Returns:
        np.ndarray: The combined audio signal with the aligned click track.
    '''
    start_frame = detect_start(signal)
    click_track_aligned = zero_pad_signal(click_track, start_frame)

    combined = None
    if pedal_reduce: 
        min_len = min(len(original_signal), len(click_track_aligned))
        combined = original_signal[:min_len] + 0.1 * click_track_aligned[:min_len]
    else:
        min_len = min(len(signal), len(click_track_aligned))
        combined = signal[:min_len] + 0.1 * click_track_aligned[:min_len]

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
    print('Generating click track plot...')
    plt.show(block=False)
    plt.pause(max_wait_time)

    return combined