#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import librosa
import accompanist

DEFAULT_CONFIG = "/Users/lukeqiao/Documents/Projects/accompanist/config/rec_tempo_detect_config.json"

def main():
    p = argparse.ArgumentParser(description="Estimate tempo and optionally play click track")
    p.add_argument("audio_file", help="Path to input audio file (absolute or relative)")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="Path to JSON config (default: repo config)")
    p.add_argument("--name", type=str, default=None, help="Name for outputs (default: audio filename stem)")
    p.add_argument("--max-wait-time", type=float, default=4.0, help="Max wait time")
    p.add_argument("--play-click", action="store_true", help="Play generated click track after processing (macOS afplay)")
    args = p.parse_args()

    audio_path = os.path.expanduser(args.audio_file)
    config_path = os.path.expanduser(args.config)

    if not os.path.exists(audio_path):
        sys.exit(f"Audio file not found: {audio_path}")
    if not os.path.exists(config_path):
        sys.exit(f"Config file not found: {config_path}")

    # Load audio from anywhere
    signal, sr = librosa.load(audio_path, sr=None)

    # Name defaults to audio filename (no extension)
    name = args.name if args.name else os.path.splitext(os.path.basename(audio_path))[0]

    # Run your pipeline
    accompanist.estimate_tempo(signal, sr, name, config_path, max_wait_time=args.max_wait_time)

    # Optionally play the click track
    if args.play_click:
        click_path = f"audio_out/click_{name}.mp3"
        if os.path.exists(click_path):
            print(f"Playing click track: {click_path}")
            try:
                subprocess.run(["afplay", click_path], check=False)
            except FileNotFoundError:
                print("afplay not found (this flag is for macOS).")
        else:
            print(f"No click track found at {click_path}")

if __name__ == "__main__":
    main()
