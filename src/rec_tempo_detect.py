import librosa
import os

import accompanist

if __name__ == "__main__":
    # load audio file
    name = 'meditation_piano_acc'
    filename = f'{name}.mp3'

    audio_path = f'audio_in/{filename}'
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f'Audio file not found: {audio_path}')
    signal, sr = librosa.load(audio_path, sr=None)

    # detect tempo
    config_file = 'config/rec_tempo_detect_config.json'
    accompanist.estimate_tempo(signal, sr, name, config_file, max_wait_time=4)
    input()
