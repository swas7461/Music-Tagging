import librosa
import numpy as np
import os
import csv
import time
from oct2py import octave
header = 'tempo beats chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()
songname = input('Enter song path: ')
y, sr = librosa.load(songname, mono=True, duration=30)
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
rmse = librosa.feature.rmse(y=y)
spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
zcr = librosa.feature.zero_crossing_rate(y)
mfcc = librosa.feature.mfcc(y=y, sr=sr)
to_append = f'{tempo} {beats.shape[0]} {np.mean(chroma_stft)} {np.mean(spec_bw)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(rolloff)} {np.mean(zcr)}'    
for e in mfcc:
    to_append += f' {np.mean(e)}'
data = to_append.split()
for i in range(0, len(data)):
    data[i] = float(data[i])
print(data)
i = int(octave.main3(data))
print(i)