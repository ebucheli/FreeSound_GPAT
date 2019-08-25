import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

import pandas as pd
import os

path_ds = '/home/edoardobucheli/Datasets/freesound-audio-tagging-2019/'
#path_train_curated = os.path.join(path_ds,'train_curated')
#df_train = pd.read_csv(os.path.join(path_ds,'train_curated.csv'))

batch_size = 64
sr=44100

def preprocess_wave(file):

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
                                samples_per_second=44100,channel_count=1)
    elif tf.__version__ == '1.14.0':
        wave,rate = tf.audio.decode_wav(file, desired_channels = 1, desired_samples = -1)

    size1 = tf.shape(wave)[0]
    diff = rate-size1
    diff = tf.cond(diff < 0, lambda:0,lambda:diff)

    pad = tf.zeros((diff,1))
    wave = tf.concat((wave,pad),axis=0)
    size2 = tf.shape(wave)[0]
    max_start = size2-rate+1

    start = tf.random_uniform((), minval=0, maxval=max_start,
                              dtype=tf.int32, seed=None, name=None)

    x = wave[start:start+44100,:]
    x = tf.expand_dims(x,axis = 0)
    x = tf.squeeze(x,axis = -1)
    x = x/tf.reduce_max(tf.abs(x),axis = 1)

    X = tf.contrib.signal.stft(x,frame_length=512,
                               frame_step = 256, fft_length = 1024)
    mX = twenty*log10(tf.abs(X))
    pX = tf.angle(X)

    mel_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(num_mel_bins = 120,
                                             num_spectrogram_bins=513,
                                             sample_rate = 44100,
                                             lower_edge_hertz=125,
                                             upper_edge_hertz=16384)
    mel_S = tf.matmul(tf.squeeze(mX,axis=0),mel_matrix)
    mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(mel_S)[..., :20]

    melmin = tf.reduce_min(mel_S)
    melmax = tf.reduce_max(mel_S)
    half_len = (melmax-melmin)/2
    melmid = melmin+half_len

    mel_S -= melmid
    mel_S /= tf.reduce_max(tf.abs(mel_S))


    return x,mX,pX,mel_S,mfccs

def load_and_preprocess_wav(path):
    file = tf.read_file(path)
    return preprocess_wave(file)

def create_and_batch_dataset_freesound(path_ds, part='curated', batch_size=32, sr=44100):

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
    wave_ds = wave_path_ds.map(load_and_preprocess_wav)
    wave_ds = wave_ds.batch(batch_size)

    return wave_ds
