# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import os
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import librosa
import librosa.display
import numpy as np
import pandas as pd
import argparse


def plot_signals_time(args):
  
    src_dir = args.src_dir
    wav_paths = glob('{}/**'.format(src_dir), recursive=True)
    wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths if '.wav' in x])
    wav_path = [x for x in wav_paths if args.fn in x]
    if len(wav_path) != 1:
        print('audio file not found for sub-string: {}'.format(args.fn))
        return
    classes = sorted(os.listdir(args.src_dir))
    real_class = int(os.path.dirname(wav_path[0]).split('/')[-1])
    titles = ['air conditioner', 'car horn', 'chiledren playing', 'dog bark',
             'drilling', 'engine idling', 'gun shot', 'jackhammer', 'siren', 'street music']
    
    print(wav_path)
    
    wav, rate = librosa.load(wav_path[0], args.sr)
    print('signal shape: ', wav.shape)
    
    librosa.display.waveplot(wav) 
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(titles[real_class])


# +
def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/50),
                       min_periods=1,
                       center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean

def test_threshold(args):
    src_dir = args.src_dir
    wav_paths = glob('{}/**'.format(src_dir), recursive=True)
    wav_path = [x for x in wav_paths if args.fn in x]
    if len(wav_path) != 1:
        print('audio file not found for sub-string: {}'.format(args.fn))
        return
    wav, rate = librosa.load(wav_path[0], args.sr)
    mask, env = envelope(wav, rate, threshold=args.threshold)
    plt.style.use('ggplot')
    plt.title('Signal Envelope, Threshold = {}'.format(str(args.threshold)))
    plt.plot(wav[mask], color='c', label='keep')
    plt.plot(wav[np.logical_not(mask)], color='r', label='remove')
    plt.plot(env, color='m', label='envelope')
    plt.grid(False)
    plt.legend(loc='best')
    plt.show()


# +
parser = argparse.ArgumentParser(description='Audio Classification Process')
parser.add_argument('--src_dir', type=str, default='../UrbanSound8K/wavfiles',
                    help='directory containing wavfiles to predict')
parser.add_argument('--fn', type=str, default='7067-6-0-0.wav',
                    help='file name to predict')
parser.add_argument('--sr', type=int, default=16000,
                    help='rate to downsample audio')
parser.add_argument('--threshold', type=str, default=0.01,
                    help='threshold magnitude for np.float32 dtype')
args, _ = parser.parse_known_args()

plot_signals_time(args)
# -

test_threshold(args)

# +
wav1, rate = librosa.load("../UrbanSound8K/clean/6/7067-6-0-0_0.wav", args.sr)
wav2, rate = librosa.load("../UrbanSound8K/clean/6/7067-6-0-0_1.wav", args.sr)

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
ax1.plot(wav1) 
ax2.plot(wav2)

# +
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

fft = np.fft.rfft(wav1)
ax1.plot(np.abs(fft)/np.sum(np.abs(fft)))
ax1.set_ylabel('Magnitude (norm)', size=18)
ax1.set_xlabel('Frequency (hertz)', size=18)

fft = np.fft.rfft(wav2)
ax2.plot(np.abs(fft)/np.sum(np.abs(fft)))
ax2.set_ylabel('Magnitude (norm)', size=18)
ax2.set_xlabel('Frequency (hertz)', size=18)

# +
n_fft = 512
hop_length = 160
n_mels = 128

wav = wav1

stft = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)
print("stft shape: ", stft.shape)

spectrogram = np.abs(stft)
print("spectrogram shape: ", spectrogram.shape)

mel = librosa.filters.mel(sr=args.sr, n_fft=n_fft, n_mels=n_mels)
print("mel filter shape: ", mel.shape)

melspectrogram = librosa.feature.melspectrogram(wav, sr=args.sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
print("melspectrogram shape: ", melspectrogram.shape)

db_melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)
print("db_melspectrogram shape: ", db_melspectrogram.shape)

plt.figure(figsize=(16,10))

plt.subplot(2, 2, 1)
librosa.display.specshow(spectrogram, sr=args.sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency(Hz)")
plt.colorbar()
plt.title("Spectrogram")

plt.subplot(2, 2, 2)
n_mels = 128
librosa.display.specshow(mel, sr=args.sr, hop_length=hop_length, x_axis='linear')
plt.ylabel("Mel filter")
plt.colorbar()
plt.title("filter bank for converting from Hz to mels.")

plt.subplot(2, 2, 3)
librosa.display.specshow(melspectrogram, sr=args.sr, hop_length=hop_length, x_axis='time', y_axis='mel');
plt.xlabel("Time")
plt.ylabel("Frequency(Hz)")
plt.colorbar()
plt.title("Melspectrogram")

plt.subplot(2, 2, 4)
librosa.display.specshow(db_melspectrogram, sr=args.sr, hop_length=hop_length, x_axis='time', y_axis='mel');
plt.xlabel("Time")
plt.ylabel("Frequency(Hz)")
plt.colorbar(format='%+2.0f dB');
plt.title("Melspectrogram in decibel")
# -


