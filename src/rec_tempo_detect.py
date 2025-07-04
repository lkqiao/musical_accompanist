import numpy as np
import librosa
import os
import matplotlib.pyplot as plt

import tempo_detect_utils

def estimate_tempo(signal, sr, name, hop_length, time_res, win_length, spp, otp, tpp, max_wait_time=15):
    # remove DC offset
    signal -= np.mean(signal)

	# preprocess signal and reduce pedal reverb
    original_signal, signal = tempo_detect_utils.damped_bandpass(signal, sr, spp['bl'], spp['bh'], pedal_reduce=spp['pedr'], 
                                                                 name=name, pedal_strength=spp['peds'])

    # get onset envelope
    onset_env = tempo_detect_utils.get_onset_env(signal, sr, hop_length, otp['lc'], otp['onth'], use_mean_onset=otp['mean'])

    # get tempogram and tempo bins
    tempogram, tempo_bins, frange = tempo_detect_utils.get_tempogram_tempo_bins(onset_env, sr, hop_length, win_length, tempo_min=otp['tmin'], tempo_max=otp['tmax'], 
                                                                                plot_tempogram=otp['plt'], name=name, time_res=time_res, max_wait_time=max_wait_time)

    # estimate tempo from tempogram
    estim_tempos, tempo_t = tempo_detect_utils.extract_tempogram_tempos(tempogram, tempo_bins, *frange, time_res, 
                                                                        peak_threshold=tpp['pth'], window_size=tpp['ws'])
    estim_tempos = tempo_detect_utils.process_tempos(estim_tempos, threshold=1)
    if tpp['ep']:
        estim_tempos = tempo_detect_utils.process_tempos(estim_tempos, threshold=0.5, step_num=2)
        estim_tempos = tempo_detect_utils.remove_spikes(estim_tempos, threshold=1)

    # plot estimated tempos and original signal
    tempo_detect_utils.plot_estim_tempos(signal, sr, estim_tempos, tempo_t, *frange, name, max_wait_time=max_wait_time/2)

    # align click track with first note
    duration = len(signal)/sr
    click_track = tempo_detect_utils.generate_click_track_from_estimates(tempo_t, estim_tempos, sr, duration)
    click_signal = tempo_detect_utils.synthesize_click_signal(signal, sr, click_track, pedal_reduce=spp['pedr'], 
                                                              original_signal=original_signal, name=name, max_wait_time=max_wait_time/2)
    return estim_tempos, tempogram, click_track, click_signal

if __name__ == "__main__":
    #  choose audio file
    name = 'meditation_piano_acc'
    filename = f'{name}.mp3'

    # load audio file
    audio_path = f'audio_in/{filename}'
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f'Audio file not found: {audio_path}')

    signal, sr = librosa.load(audio_path, sr=None)  

    print('----------------- Signal and Tempo Detection Stats -----------------')
    print(f'Sample rate: {sr}')
    print(f'Signal length: {len(signal)}')
    print(f'Signal duration: {len(signal)/sr:.3f} s')

    # ----------------------------------------------------------------------------------------------
    # define parameters
    hop_length = 256  # distance between frames
    win_dur = 1.5  # window duration in seconds (use 4 for percussive pieces, <= 2 for classical pieces)

    time_res = hop_length/sr  # temporal resolution
    freq_res = 1/win_dur  # frequency resolution
    win_length = int((win_dur*sr)/hop_length)  # window length in frames
    print(f'Temporal resolution: {time_res:.3f} s')
    print(f'Frequency resolution: {freq_res:.3f} Hz/bin')
    print(f'Window length: {win_length} frames')

    # for preprocessing
    signal_preprocessing_params = {
        'pedr': True,  # pedal_reduce
        'peds': 0.5,  # pedal_strength
        'bh': 4000,  # bandpass_high
        'bl': 20  # bandpass_low
    }

    # for onset envelope, tempogram, and tempo bins 
    onset_tempogram_params = {
        'lc': 10000,  # lowpass_cutoff
        'tmin': 35,  # tempo_min
        'tmax': 200,  # tempo_max
        'onth': 0.075,  # onset_threshold
        'mean': False,  # use_mean_onset
        'plt': True  # plot_tempogram
    }

    # for tempo estimation postprocessing
    tempo_postprocessing_params = {
        'pth': 0.2,  # peak_threshold
        'ws': 5,  # window_size
        'ep': True  # extra_processing
    }
    # ----------------------------------------------------------------------------------------------

    estimate_tempo(signal, sr, name, hop_length, time_res, win_length, signal_preprocessing_params,
                   onset_tempogram_params, tempo_postprocessing_params, max_wait_time=4)
    input()