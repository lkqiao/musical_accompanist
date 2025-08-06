import numpy as np
import json

import tempo_detect_utils

def estimate_tempo(signal, sr, name, config_file, max_wait_time=15):
    '''
    Estimate the tempo of an audio signal using onset envelope and tempogram analysis.

    This function processes an input audio signal to estimate its tempo over time. It removes DC offset, applies bandpass 
    filtering to reduce pedal reverb, computes the onset envelope, and generates a tempogram to extract tempo estimates. 
    Optionally, it processes the estimated tempos to remove spikes and smooth the results. The function also generates and 
    aligns a click track with the detected tempo and synthesizes a corresponding click signal.

    Parameters:
        signal (np.ndarray): The input audio signal.
        sr (int): Sampling rate of the audio signal.
        name (str): Name identifier for the current processing (used for plotting and saving).
        config_file (str): Path to a JSON config file containing parameters for tempo detection.
        max_wait_time (float, optional): Maximum wait time for plotting and processing (in seconds). Default is 15.

    Returns:
        tuple:
            estim_tempos (np.ndarray): Estimated tempo values over time.
            tempogram (np.ndarray): Computed tempogram matrix.
            click_track (np.ndarray): Generated click track aligned with estimated tempo.
            click_signal (np.ndarray): Synthesized audio signal of the click track.
    '''
    with open(config_file, 'r') as f:
        config = json.load(f)

    hop_length = config['hop_length']
    win_dur = config['win_dur']
    signal_preprocessing_params = config['signal_preprocessing_params']
    onset_tempogram_params = config['onset_tempogram_params']
    tempo_postprocessing_params = config['tempo_postprocessing_params']

    time_res = hop_length/sr
    win_length = int((win_dur*sr)/hop_length)

    # remove DC offset
    signal -= np.mean(signal)

    # preprocess signal and reduce pedal reverb
    original_signal, signal = tempo_detect_utils.damped_bandpass(
        signal, sr, signal_preprocessing_params['bl'], signal_preprocessing_params['bh'], 
        pedal_reduce=signal_preprocessing_params['pedr'], name=name, 
        pedal_strength=signal_preprocessing_params['peds']
    )

    # get onset envelope
    onset_env = tempo_detect_utils.get_onset_env(
        signal, sr, hop_length, onset_tempogram_params['lc'], onset_tempogram_params['onth'],
        use_max=onset_tempogram_params['max'], mean_max_weight=onset_tempogram_params['mmw']
    )

    # get tempogram and tempo bins
    tempogram, tempo_bins, frange = tempo_detect_utils.get_tempogram_tempo_bins(
        onset_env, sr, hop_length, win_length, tempo_min=onset_tempogram_params['tmin'], 
        tempo_max=onset_tempogram_params['tmax'], plot_tempogram=onset_tempogram_params['plt'], 
        name=name, time_res=time_res, max_wait_time=max_wait_time
    )

    # estimate tempo from tempogram
    estim_tempos, tempo_t = tempo_detect_utils.extract_tempogram_tempos(
        tempogram, tempo_bins, *frange, time_res, peak_threshold=tempo_postprocessing_params['pth'], 
        window_size=tempo_postprocessing_params['ws']
    )
    estim_tempos = tempo_detect_utils.process_tempos(estim_tempos, threshold=1)
    if tempo_postprocessing_params['ep']:
        estim_tempos = tempo_detect_utils.process_tempos(estim_tempos, threshold=0.5, step_num=2)
        estim_tempos = tempo_detect_utils.remove_spikes(estim_tempos, threshold=1)

    # plot estimated tempos and original signal
    tempo_detect_utils.plot_estim_tempos(signal, sr, estim_tempos, tempo_t, *frange, name, max_wait_time=max_wait_time/2)

    # align click track with first note
    duration = len(signal)/sr
    click_track = tempo_detect_utils.generate_click_track_from_estimates(tempo_t, estim_tempos, sr, duration)
    click_signal = tempo_detect_utils.synthesize_click_signal(
        signal, sr, click_track, pedal_reduce=signal_preprocessing_params['pedr'],
        original_signal=original_signal, name=name, max_wait_time=max_wait_time/2
    )
    return estim_tempos, tempogram, click_track, click_signal
