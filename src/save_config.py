import json    
import librosa
import os

def convert(obj):
    '''
    Convert any tuples to lists for JSON serialization.

    Parameters:
        obj (any): The object to convert.

    Returns:
        any: The converted object with tuples replaced by lists.
    '''
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    return obj

if __name__ == "__main__":
    #  choose audio file
    name = 'meditation_piano_acc'
    filename = f'{name}.mp3'

    # load audio file
    audio_path = f'audio_in/{filename}'
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f'Audio file not found: {audio_path}')

    _, sr = librosa.load(audio_path, sr=None)  

    # define parameters
    hop_length = 256  # distance between frames
    win_dur = 1.5  # window duration in seconds (use 4 for percussive pieces, <= 2 for classical pieces)

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
        'max': True,  # use_max
        'mmw': (0.3,0.7),  # mean_max_weight
        'plt': True  # plot_tempogram
    }

    # for tempo estimation postprocessing
    tempo_postprocessing_params = {
        'pth': 0.2,  # peak_threshold
        'ws': 5,  # window_size
        'ep': True  # extra_processing
    }

    # collect all parameters into a dictionary
    config = {
        'hop_length': hop_length,
        'win_dur': win_dur,
        'signal_preprocessing_params': signal_preprocessing_params,
        'onset_tempogram_params': onset_tempogram_params,
        'tempo_postprocessing_params': tempo_postprocessing_params
    }

    config_json = convert(config)

    # save to json file
    with open('src/rec_tempo_detect_config.json', 'w') as f:
        json.dump(config_json, f, indent=4)
