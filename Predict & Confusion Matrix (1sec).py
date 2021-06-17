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

# +
from tensorflow.keras.models import load_model
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from tensorflow.keras.utils import to_categorical
import numpy as np
from glob import glob
import argparse
import os
import pandas as pd
from tqdm import tqdm
import librosa
import matplotlib.pyplot as plt
import pylab
import seaborn as sns

'''
Make predictions for multiple files
'''    
def make_predictions(args):

    model = load_model(args.model_fn,
        custom_objects={'STFT':STFT,
                        'Magnitude':Magnitude,
                        'ApplyFilterbank':ApplyFilterbank,
                        'MagnitudeToDecibel':MagnitudeToDecibel})
    wav_paths = glob('{}/**'.format(args.src_dir), recursive=True)
    wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths if '.wav' in x])
    classes = sorted(os.listdir(args.src_dir))
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    le = LabelEncoder()
    y_true = le.fit_transform(labels)
    results = []

    for z, wav_fn in enumerate(wav_paths):
        wav, rate = librosa.load(wav_fn, args.sr)
        wav = wav.reshape(-1,1)
        batch = []
        batch.append(wav)
        X_batch = np.array(batch, dtype=np.float32)
        y_pred = model.predict(X_batch)
        y_pred = np.argmax(y_pred)
        
        real_class = os.path.dirname(wav_fn).split('/')[-1]
        print('Actual class: {}, Predicted class: {}'.format(real_class, classes[y_pred]))
        results.append(y_pred)

    np.save(os.path.join('../Thingy52/combined/logs', args.pred_fn), np.array(results))


# -

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--model_fn', type=str, default='../Thingy52/combined/models/conv1d.h5',
                        help='model file to make predictions')
    parser.add_argument('--pred_fn', type=str, default='conv1d_y_pred',
                        help='fn to write predictions in logs dir')
    parser.add_argument('--src_dir', type=str, default='../Thingy52/combined/validation',
                        help='directory containing wavfiles to predict')
    parser.add_argument('--dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='sample rate of clean audio')
    parser.add_argument('--threshold', type=str, default=0.00,
                        help='threshold magnitude for np.int16 dtype')
    args, _ = parser.parse_known_args()

    make_predictions(args)

# +
# Confusion Matrix
y_pred = np.load('../Thingy52/combined/logs/conv1d_y_pred.npy')

wav_paths = glob('{}/**'.format('../Thingy52/combined/validation'), recursive=True)
wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths if '.wav' in x])
classes = sorted(os.listdir('../Thingy52/combined/validation'))
labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
unique_labels = np.unique(labels)
name_labels = ['speech','tv','drop','door','water','toilet','brush teeth','vacuum','chop','fry/boil','dish']
le = LabelEncoder()
y_true = le.fit_transform(labels)
y_pred_labels = [classes[x] for x in y_pred]
y_true_labels = [classes[x] for x in y_true]

conf_mat = confusion_matrix(y_true_labels, y_pred_labels, labels=unique_labels, normalize = 'true')
df_cm = pd.DataFrame(conf_mat, index = name_labels,
                     columns = name_labels)
plt.figure(figsize = (10,8))
plt.title('Confusion Matrix')
sns.heatmap(df_cm, annot=True, cmap='viridis')
plt.show()
# -

