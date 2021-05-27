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
'''
group files in different folders in same class directory
'''
import os
import fnmatch
import shutil

for dirpath, dirnames, files in os.walk('../Thingy52/audio'):
    for filename in files:
        for i in range(0,10):
            if fnmatch.fnmatch(filename, '*-*-*-'+str(i)+'.wav'):
                shutil.copy2(dirpath+'/'+filename, '../Thingy52/wavfiles/'+str(i)+'/')

# +
'''
Functions used for processing audio files
'''
import matplotlib.pyplot as plt
from scipy.io import wavfile
import argparse
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import wavio
import librosa


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/20),
                       min_periods=1,
                       center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean


def save_sample(sample, rate, target_dir, fn, ix):
    fn = fn.split('.wav')[0]
    dst_path = os.path.join(target_dir, fn+'_{}.wav'.format(str(ix)))
    print(dst_path)
    if os.path.exists(dst_path):
        return
    wavfile.write(dst_path, rate, sample)


def check_dir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)

# split and organize DESED dataset into classes
def desed(args):
    desed_root = args.desed_root
    desed_label_root = args.desed_label_root
    
    desed_label = pd.read_csv(desed_label_root, delimiter='\t')
    desed_label = desed_label.dropna(axis=0)
    desed_class = ['Speech', 'Dog', 'Dishes', 'Alarm_bell_ringing', 'Cat', 'Running_water', 
                   'Blender', 'Frying', 'Vacuum_cleaner', 'Electric_shaver_toothbrush']
    for _cls in desed_class:
        check_dir(desed_root+'/../'+_cls)
    
    for fn in tqdm(desed_label.index):
        event_label = desed_label.event_label[fn]
        offset = desed_label.onset[fn]
        duration = desed_label.offset[fn] - desed_label.onset[fn]
        dst_path = os.path.join(desed_root,'..',event_label,str(fn)+'.wav')

        if os.path.exists(desed_root+'/'+desed_label.filename[fn]):
            wav, rate = librosa.load(desed_root+'/'+desed_label.filename[fn], sr=args.sr, offset=offset, duration=duration)
            wavfile.write(dst_path, rate, wav)

def split_wavs(args):
    src_root = args.src_root
    dst_root = args.dst_root
    dt = args.delta_time

    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_paths = [x for x in wav_paths if '.wav' in x]
    dirs = os.listdir(src_root)
    check_dir(dst_root)
    classes = os.listdir(src_root)
    for _cls in classes:
        target_dir = os.path.join(dst_root, _cls)
        check_dir(target_dir)
        src_dir = os.path.join(src_root, _cls)
        for fn in tqdm(os.listdir(src_dir)):
            src_fn = os.path.join(src_dir, fn)
            wav, rate = librosa.load(src_fn, args.sr)
            mask, y_mean = envelope(wav, rate, threshold=args.threshold)
            wav = wav[mask]
            delta_sample = int(dt*rate)

            '''
            cleaned audio is less than a single sample
            pad with zeros to delta_sample size
            '''
            if wav.shape[0] < delta_sample:
                sample = np.zeros(shape=(delta_sample,), dtype=np.int16)
                sample[:wav.shape[0]] = wav
                save_sample(sample, rate, target_dir, fn, 0)
            # step through audio and save every delta_sample
            # discard the ending audio if it is too short
            else:
                trunc = wav.shape[0] % delta_sample
                for cnt, i in enumerate(np.arange(0, wav.shape[0]-trunc, delta_sample)):
                    start = int(i)
                    stop = int(i + delta_sample)
                    sample = wav[start:stop]
                    save_sample(sample, rate, target_dir, fn, cnt)


def test_threshold(args):
    src_root = args.src_root
    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_path = [x for x in wav_paths if args.fn in x]
    if len(wav_path) != 1:
        print('audio file not found for sub-string: {}'.format(args.fn))
        return
    wav, rate = librosa.load(wav_path[0], args.sr)
    mask, env = envelope(wav, rate, threshold=args.threshold)
    plt.style.use('ggplot')
    plt.title('Signal Envelope, Threshold = {}'.format(str(args.threshold)))
    plt.plot(wav[mask], color='c', label='keep')
    plt.plot(env, color='m', label='envelope')
    plt.plot(wav[np.logical_not(mask)], color='r', label='remove')
    plt.grid(False)
    plt.legend(loc='best')
    plt.show()


# -

'''
set main arguments 
test and set threshold
split audio files in samples with delta time 
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Cleaning audio data')
    parser.add_argument('--src_root', type=str, default='../DESED/data/desed_labeled/soundbank_foreground/train',
                        help='directory of audio files in total duration')
    parser.add_argument('--dst_root', type=str, default='../DESED/data/desed_labeled/soundbank_foreground/train_clean',
                        help='directory to put audio files split by delta_time')
    parser.add_argument('--delta_time', '-dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='rate to downsample audio')

    parser.add_argument('--fn', type=str, default='210410-174729-712-719-2.wav',
                        help='file to plot over time to check magnitude')
    parser.add_argument('--threshold', type=str, default=0.003,
                        help='threshold magnitude for np.float32 dtype')
    
    parser.add_argument('--desed_root', type=str, default='../DESED/data/dcase21_synth/audio/evaluation/synthetic21_evaluation/soundscapes',
                        help='directory of desed real audio files')
    parser.add_argument('--desed_label_root', type=str, default='../DESED/data/dcase21_synth/metadata/evaluation/synhtetic21_evaluation/soundscapes.tsv',
                        help='directory and filename of desed metadata label')
    args, _ = parser.parse_known_args()

    #test_threshold(args)
    split_wavs(args)
    #desed(args)


# +
'''
count sample files in each class folders
'''
count = 0
for i in range(0,10):
    list = os.listdir("../Thingy52/clean/" + str(i)) # dir is your directory path
    print("../Thingy52/clean/" + str(i) + ": " + str(len(list)))
    count = count + len(list)

print("Total sample files: " + str(count))
# +
'''
count sample files in each class folders
'''
count = 0
desed_class = ['Dishes', 'Running_water', 'Frying', 'Vacuum_cleaner']
for i in desed_class:
    list = os.listdir("../DESED/data/desed_labeled/test/data/real/" + i) # dir is your directory path
    print(i + ": " + str(len(list)))
    count = count + len(list)

print("Total sample files: " + str(count))


# -





