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

import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization
import kapre
from tensorflow.keras.models import Sequential
from kapre.composed import get_melspectrogram_layer, get_log_frequency_spectrogram_layer
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io import wavfile


def visualise_model(X):
    fig, ax = plt.subplots(figsize=(10,8))
    ax.set_title('Normalized Frequency Spectrogram', size=20)
    ax.imshow(X)
    ax.set_ylabel('Mel bins', size=18)
    ax.set_xlabel('Time (10 ms)', size=18)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    plt.show()


# +
src, sr = librosa.load('../Thingy52/clean/0/210410-174725-31-40-0_4.wav', sr=None, mono=True)
print('Audio length: %d samples, %04.2f seconds. \n' % (len(src), len(src) / sr) +
      'Audio sample rate: %d Hz' % sr)
dt = 1.0

D = librosa.amplitude_to_db(np.abs(librosa.stft(src)), ref=np.max)
librosa.display.specshow(D, y_axis='linear', x_axis='time',sr=sr)

_src = src[:int(sr*dt)]
src = np.expand_dims(_src, axis=1)
input_shape = src.shape
print(input_shape)

# +
hop_length = 1024
src, sr = librosa.load('../Thingy52/wavfiles/0/210410-174725-31-40-0.wav', sr=None, mono=True)
print('Audio length: %d samples, %04.2f seconds. \n' % (len(src), len(src) / sr) +
      'Audio sample rate: %d Hz' % sr)
dt = 1.0

D = librosa.amplitude_to_db(np.abs(librosa.stft(src, hop_length=hop_length)), ref=np.max)
librosa.display.specshow(D, y_axis='linear', x_axis='time',sr=sr)

_src = src[:int(sr*dt)]
src = np.expand_dims(_src, axis=1)
input_shape = src.shape
print(input_shape)
# -

melgram = get_melspectrogram_layer(input_shape=input_shape,
                                   n_mels=128,
                                   mel_norm='slaney',
                                   pad_end=True,
                                   n_fft=512,
                                   win_length=400,
                                   hop_length=160,
                                   sample_rate=sr,
                                   db_ref_value=1.0,
                                   return_decibel=True,
                                   input_data_format='channels_last',
                                   output_data_format='channels_last')
norm = LayerNormalization(axis=2)
melgram.shape = (16000, 1)
model = Sequential()
model.add(melgram)
model.add(norm)
model.summary()

# +
batch = np.expand_dims(src, axis=0)
X = model.predict(batch).squeeze().T
visualise_model(X)

plt.title('Normalized Frequency Histogram')
plt.hist(X.flatten(), bins='auto')
plt.show()
# -


