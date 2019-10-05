import tensorflow as tf
import numpy as np

from CNNetworks2D import malley_cnn_80, malley_cnn_120
from CNNetworks1D import conv1d_v1
from RNNetworks import AttRNNSpeechModel

def load_network(network,input_shape,n_classes,lr,weights = None, new_head = False, train_only_head = False):

    if network == 'malley':

        if new_head:

            assert weights is not None, 'Must provide weights if new_head = True'

            if input_shape[0] == 80:
                model = malley_cnn_80(input_shape,41)

            elif input_shape[0] == 120:
                model = malley_cnn_120(input_shape,41)

            model.load_weights(weights)


            new_output_name = 'cut_here'
            new_output_layer = model.get_layer(new_output_name).output
            model_headless = tf.keras.Model(inputs = model.input, outputs = new_output_layer)

            X = tf.keras.layers.Dense(512,activation='relu')(model_headless.output)
            X = tf.keras.layers.Dropout(0.5)(X)
            X = tf.keras.layers.Dense(n_classes, activation = 'softmax')(X)

            model = tf.keras.Model(inputs = model_headless.input, outputs = X)

            if train_only_head:
                for l in model.layers[:13]:
                    l.trainable = False

        else:
            if input_shape[0] == 80:
                model = malley_cnn_80(input_shape,n_classes)

            elif input_shape[0] == 120:
                model = malley_cnn_120(input_shape,n_classes)

            if weights is not None:
                model.load_weights(weights)


    elif network == 'CNN1D':

        model = conv1d_v1(input_shape,n_classes)

        if weights is not None:
            model.load_weights(weights)

        if train_only_head:
            new_output_name = 'cut_here'
            new_output_layer = model.get_layer(new_output_name).output
            model_headless = tf.keras.Model(inputs = model.input, outputs = new_output_layer)

            X = tf.keras.layers.Dense(64,activation='relu')(model_headless.output)
            X = tf.keras.layers.Dense(128,activation='relu')(X)
            X = tf.keras.layers.Dense(n_classes, activation = 'softmax')(X)

            model = tf.keras.Model(inputs = model_headless.input, outputs = X)

            for l in model.layers[:16]:
                l.trainable = False

    elif network == 'AttRNN2D':

        model = AttRNNSpeechModel(input_shape,n_classes)

        if weights is not None:
            model.load_weights(weights)

    else:
        print('Invalid network selection')
        exit()

    return model
