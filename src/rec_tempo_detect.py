import numpy as np
import matplotlib.pyplot as plt
import librosa
from tempocnn.classifier import TempoClassifier
from tempocnn.feature import read_features

# params
model_name = 'cnn'
input_file = 'audio_tests/minecraft.mp3'
frames = 256
hop_length = 128
sample_rate = 44100

# load audio
y, sr = librosa.load(input_file, sr=sample_rate)
duration = len(y) / sr  # in seconds
print(f"Audio duration: {duration:.2f} seconds")

# tempo estimation
classifier = TempoClassifier(model_name)
features = read_features(input_file, frames=frames, hop_length=hop_length)
local_tempo_classes = classifier.estimate(features)
max_predictions = np.argmax(local_tempo_classes, axis=1)
local_tempi = classifier.to_bpm(max_predictions)
print(f"Estimated tempo: {local_tempi}")

# time axis
num_tempi = len(local_tempi)
times = np.linspace(0, duration, num=num_tempi)

# plotting
plt.figure(figsize=(10, 4))
plt.plot(times, local_tempi, label="Estimated Tempo (BPM)", linewidth=1.5)
plt.xlabel("Time (s)")
plt.ylabel("Tempo (BPM)")
plt.title("Tempo vs Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()