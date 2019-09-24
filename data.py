import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

import pandas as pd
import os

from random import shuffle

path_ds = '/home/edoardobucheli/Datasets/freesound-audio-tagging-2019/'
#path_train_curated = os.path.join(path_ds,'train_curated')
#df_train = pd.read_csv(os.path.join(path_ds,'train_curated.csv'))

batch_size = 64
sr=44100

def preprocess_wave(file,
                    transformation,
                    normalize,
                    sr,
                    len_sec,
                    frame_length,
                    frame_step,
                    fft_length,
                    n_mels,
                    n_mfcc,
                    mel_lower_edge,
                    mel_upper_edge):

    def log10(x):
        numerator = tf.log(x)
        denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

    twenty = tf.constant(20,dtype = tf.float32)
    zero = tf.constant(0,dtype = tf.float32)
    one = tf.constant(1,dtype = tf.int32)
    zeroint = tf.constant(0,dtype = tf.int32)

    if tf.__version__ == '1.12.0':
        rate = tf.constant(44100, dtype = tf.int32)
        wave = tf.contrib.ffmpeg.decode_audio(file,file_format='wav',
                                samples_per_second=sr,channel_count=1)
    elif tf.__version__ == '1.14.0':
        wave,rate = tf.audio.decode_wav(file, desired_channels = 1, desired_samples = -1)

    duration = tf.constant(sr*len_sec,tf.int32)

    size1 = tf.shape(wave)[0]
    #diff = rate-size1
    diff = duration-size1
    diff = tf.cond(diff < 0, lambda:0,lambda:diff)

    pad = tf.zeros((diff,1))
    wave = tf.concat((wave,pad),axis=0)
    size2 = tf.shape(wave)[0]
    max_start = size2-duration+1

    start = tf.random_uniform((), minval=0, maxval=max_start,
                              dtype=tf.int32, seed=None, name=None)

    x = wave[start:start+duration,:]
    x = tf.expand_dims(x,axis = 0)
    x = tf.squeeze(x,axis = -1)
    x = x/tf.reduce_max(tf.abs(x),axis = 1)

    if transformation == 'wave':
        rep = x

    else:

        X = tf.contrib.signal.stft(x,
                                   frame_length = frame_length,
                                   frame_step = frame_step,
                                   fft_length = fft_length)

    #pX = tf.angle(X)

    if transformation == 'mag':
        rep = twenty*log10(tf.abs(X))

        if normalize:
            mXmax = tf.reduce_max(rep)
            rep = tf.where(tf.is_inf(rep),-rep,rep)
            mXmin = tf.reduce_min(rep)

            rep = tf.where(tf.is_inf(rep),tf.ones_like(rep)*mXmin,rep)

            mX_half_len = (mXmax-mXmin)/2
            mXmid = mXmin+mX_half_len

            rep -= mXmid
            rep /= tf.reduce_max(tf.abs(rep))

    elif transformation == 'mel':
        mel_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(num_mel_bins = n_mels,
                                                 num_spectrogram_bins = int(fft_length/2+1),
                                                 sample_rate = sr,
                                                 lower_edge_hertz = mel_lower_edge,
                                                 upper_edge_hertz = mel_upper_edge)

        mX = twenty*log10(tf.abs(X))
        rep = tf.matmul(tf.squeeze(mX,axis=0),mel_matrix)

        if normalize:
            melmin = tf.reduce_min(rep)
            melmax = tf.reduce_max(rep)
            half_len = (melmax-melmin)/2
            melmid = melmin+half_len

            rep -= melmid
            rep /= tf.reduce_max(tf.abs(rep))

            rep = tf.where(tf.is_nan(rep),tf.ones_like(rep)*-1,rep)

    elif transformation == 'mfcc':
        mel_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(num_mel_bins = n_mels,
                                                 num_spectrogram_bins = int(fft_length/2+1),
                                                 sample_rate = sr,
                                                 lower_edge_hertz = mel_lower_edge,
                                                 upper_edge_hertz = mel_upper_edge)
        mX = twenty*log10(tf.abs(X))
        mel_S = tf.matmul(tf.squeeze(mX,axis=0),mel_matrix)
        rep = tf.contrib.signal.mfccs_from_log_mel_spectrograms(mel_S)[..., :n_mfcc]

    return rep

def load_and_preprocess_wav(path,
                            transformation,
                            normalize,
                            sr,
                            len_sec,
                            frame_length,
                            frame_step,
                            fft_length,
                            n_mels,
                            n_mfcc,
                            mel_lower_edge,
                            mel_upper_edge):

    _preprocess_wave = lambda _file: preprocess_wave(_file,
                                                     transformation,
                                                     normalize,
                                                     sr,
                                                     len_sec,
                                                     frame_length,
                                                     frame_step,
                                                     fft_length,
                                                     n_mels,
                                                     n_mfcc,
                                                     mel_lower_edge,
                                                     mel_upper_edge)

    file = tf.read_file(path)
    return _preprocess_wave(file)

def create_and_batch_dataset_freesound(path_ds, part='curated',
                                       batch_size=32,
                                       transformation = 'mag',
                                       normalize = True,
                                       sr=44100,
                                       len_sec=1,
                                       frame_length=512,
                                       frame_step = 256,
                                       fft_length = 1024,
                                       n_mels = 40,
                                       n_mfcc = 40,
                                       mel_lower_edge=125,
                                       mel_upper_edge=16384):

    if part == 'curated':
        df = pd.read_csv(os.path.join(path_ds,'train_curated.csv'))
        path_ds = os.path.join(path_ds,'train_curated')

        names = df['fname'].tolist()
        names_paths = [os.path.join(path_ds,f) for f in names]
        print(len(names))
        labels = df['labels'].tolist()

    elif part == 'noisy':
        df = pd.read_csv(os.path.join(path_ds,'train_noisy.csv'))
        path_ds = os.path.join(path_ds,'train_noisy')

        names = df['fname'].tolist()
        names_paths = [os.path.join(path_ds,f) for f in names]
        print(len(names))
        labels = df['labels'].tolist()

    wave_path_ds = tf.data.Dataset.from_tensor_slices(names_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)

    load_and_preprocess_rep = lambda _file: load_and_preprocess_wav(_file,
                                            transformation,
                                            normalize,
                                            sr,
                                            len_sec,
                                            frame_length,
                                            frame_step,
                                            fft_length,
                                            n_mels,
                                            n_mfcc,
                                            mel_lower_edge,
                                            mel_upper_edge)

    wave_ds = wave_path_ds.map(load_and_preprocess_rep)

    wave_label_ds = tf.data.Dataset.zip((wave_ds,label_ds))
    wave_label_ds = wave_label_ds.batch(batch_size)

    return wave_label_ds

def create_and_batch_dataset_TFSSC(path_ds, part='train',
                                   batch_size=64,
                                   transformation = 'mag',
                                   normalize = True,
                                   sr=16000,
                                   len_sec=1,
                                   frame_length=256,
                                   frame_step = 256,
                                   fft_length = 256,
                                   n_mels = 40,
                                   n_mfcc = 40,
                                   mel_lower_edge=125,
                                   mel_upper_edge=8000):


    word_to_label = {'yes':0,'no':1,'up':2,'down':3,'left':4,'right':5,
                 'on':6,'off':7,'stop':8,'go':9,
                 'backward':10, 'bed':10,'bird':10,'cat':10,'dog':10,
                 'follow':10,'forward':10,'happy':10,'house':10,'learn':10,
                 'marvin':10,'sheila':10,'tree':10,'visual':10,'wow':10,
                 'zero':10,'one':10,'two':10,'three':10,'four':10,
                 'five':10,'six':10,'seven':10,'eight':10,'nine':10}

    label_to_word = dict([[v,k] for k,v in word_to_label.items()])
    label_to_word[10] = '<unk>'

    if part == 'train':

        with open(os.path.join(path_ds,'Partitions/10Words/training_files.txt'),'r') as f:
            names = f.read().splitlines()

        shuffle(names)

        names_paths = [os.path.join(path_ds,'audio',f) for f in names]
        print('Found {} files in partition'.format(len(names)))
        labels = [f.split('/')[0] for f in names]

    elif part == 'validation':
        with open(os.path.join(path_ds,'Partitions/10Words/validation_files.txt'),'r') as f:
            names = f.read().splitlines()
        shuffle(names)
        names_paths = [os.path.join(path_ds,'audio',f) for f in names]
        print('Found {} files in partition'.format(len(names)))
        labels = [f.split('/')[0] for f in names]

    elif part == 'test':
        with open(os.path.join(path_ds,'Partitions/10Words/testing_files.txt'),'r') as f:
            names = f.read().splitlines()
        shuffle(names)
        names_paths = [os.path.join(path_ds,'audio',f) for f in names]
        print('Found {} files in partition'.format(len(names)))
        labels = [f.split('/')[0] for f in names]

    labels_int = [word_to_label[f] for f in labels]

    wave_path_ds = tf.data.Dataset.from_tensor_slices(names_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels_int)

    load_and_preprocess_rep = lambda _file: load_and_preprocess_wav(_file,
                                            transformation,
                                            normalize,
                                            sr,
                                            len_sec,
                                            frame_length,
                                            frame_step,
                                            fft_length,
                                            n_mels,
                                            n_mfcc,
                                            mel_lower_edge,
                                            mel_upper_edge)

    wave_ds = wave_path_ds.map(load_and_preprocess_rep)

    wave_label_ds = tf.data.Dataset.zip((wave_ds,label_ds))
    wave_label_ds = wave_label_ds.batch(batch_size)
    wave_label_ds = wave_label_ds.prefetch(buffer_size = 1024)
    #wave_label_ds = wave_label_ds.shuffle(len(labels))

    return wave_label_ds
