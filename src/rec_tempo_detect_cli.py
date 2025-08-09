#!/usr/bin/env python3
import sys
import os
import librosa
import accompanist

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: accomp-tempo <audio_file> <config_file> [--max-wait-time N]")
        sys.exit(1)

    audio_path = sys.argv[1]
    config_file = sys.argv[2]
    max_wait_time = 4

    # Optional flag
    if len(sys.argv) > 3 and sys.argv[3] == "--max-wait-time":
        if len(sys.argv) > 4:
            max_wait_time = float(sys.argv[4])
        else:
            print("Error: Missing value for --max-wait-time")
            sys.exit(1)

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    signal, sr = librosa.load(audio_path, sr=None)
    name = os.path.splitext(os.path.basename(audio_path))[0]
    accompanist.estimate_tempo(signal, sr, name, config_file, max_wait_time=max_wait_time)
