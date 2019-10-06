import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from copy import deepcopy

import os
import pickle
import argparse
import json

from utilities import get_all_classes_dict, get_classes_to_meta_dict, get_labels
from utilities import plot_cm, create_quick_test_2d,create_quick_test_wave, get_x_and_labels
from data_generator import DataGenerator, DataGeneratorWave
from network_loader import load_network

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
# representation
parser.add_argument('--sr', dest = 'sr', type = int,default = 16000,
                    help = 'Choose Sampling Rate, either 16000 or 32000.')
parser.add_argument('--hop_length', dest = 'hop_length', type = int, default = 512,
                    help = 'Hop Length for FFT choose 128, 256 or 512.')
parser.add_argument('--freq_res', dest = 'freq_res',type = int,default = 80,
                    help = 'Choose frequency resolution for MS, MFCC or PS')
parser.add_argument('--representation', dest = 'representation', default = 'MS',
                    help = 'Choose representation: PS, MS or MFCC')
# model
parser.add_argument('--pt_weights', dest = 'pt_weights', default = None,
                    help = 'Specify location of pretrained weights, must match the selected model and representation')
parser.add_argument('--network', dest = 'network', default = 'malley',
                    help = 'Choose network: malley, cnn1d, attrnn1d, attrnn2d')
parser.add_argument('--problem', dest = 'problem', default = 'Cluster',
                    help = 'Choose problem: 41C, MC, or Cluster')
parser.add_argument('--cluster', dest = 'cluster', type = int, default = 0,
                    help = 'Choose cluster if Cluster for --problem')
# training
parser.add_argument('--use_only_curated', dest = 'use_only_curated', action = 'store_true')
parser.add_argument('--epochs', dest = 'epochs', type = int,default = 40)
parser.add_argument('--lr', dest = 'lr', type = float, default = 0.001)
parser.add_argument('--batch_size', dest = 'batch_size', type = int, default = 128)

args = parser.parse_args()

#def train_model(sr, hop_length,freq_res, representation,pt_weights, network, problem,cluster,use_only_curated,epochs,lr,batch_size):
def train_model(args):

    sr = args.sr
    hop_length = args.hop_length
    freq_res = args.freq_res
    representation = args.representation

    pt_weights = args.pt_weights
    network = args.network
    problem = args.problem
    cluster = args.cluster

    use_only_curated = args.use_only_curated
    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size
    log = vars(args)

    path_dataset = '/home/edoardobucheli/Datasets/FSDKaggle2018'

    if sr == 16000:
        sample_rate = '16k'
        path_train = os.path.join(path_dataset,'audio_train_16k')
        path_test = os.path.join(path_dataset,'audio_test_16k')
    elif sr == 32000:
        sample_rate = '32k'
        path_train = os.path.join(path_dataset,'audio_train_32k')
        path_test = os.path.join(path_dataset,'audio_test_32k')
    else:
        print("Sample Rate option not available")
        exit()

    # Load label data
    train_data = pd.read_csv(os.path.join(path_dataset,'train_post_competition.csv'))
    test_data = pd.read_csv(os.path.join(path_dataset,'test_post_competition_scoring_clips.csv'))

    num_to_label, label_to_num, n_classes = get_all_classes_dict(train_data)
    label_to_meta, label_num_to_meta = get_classes_to_meta_dict(label_to_num)

    data_cur = train_data[train_data['manually_verified']==1]
    data_noi = train_data[train_data['manually_verified']==0]

    meta_labels_all, labels_all = get_labels(train_data,label_to_meta, label_to_num)
    meta_labels_cur, labels_cur = get_labels(data_cur,label_to_meta, label_to_num)
    meta_labels_noi, labels_noi = get_labels(data_noi,label_to_meta, label_to_num)
    meta_labels_test, labels_test = get_labels(test_data,label_to_meta, label_to_num)

    n_meta_classes = len(np.unique(meta_labels_all))

    # Load Data

    file_length = 64000
    frames = int(np.ceil(file_length/hop_length))

    if representation == 'WF':
        experiment_name = '{}-{}-{}'.format(network,sample_rate,representation)

        pickle_train = './preprocessed_train/{}-{}-64k'.format(representation,sample_rate)
        pickle_test = './preprocessed_test/{}-{}-64k'.format(representation,sample_rate)
        input_shape = [file_length,]
    else:
        experiment_name = '{}-{}-{}-{}-HL{}'.format(network,sample_rate,representation,freq_res,hop_length)

        pickle_train = './preprocessed_train/{}-{}-HL{}-WF{}-64k'.format(representation,
                                                                         freq_res,hop_length,
                                                                         sample_rate)
        pickle_test = './preprocessed_test/{}-{}-HL{}-WF{}-64k'.format(representation,
                                                                         freq_res,hop_length,
                                                                         sample_rate)
        input_shape = [freq_res,frames]


    with open(pickle_train,'rb') as fp:
        x_train = pickle.load(fp)
    with open(pickle_test, 'rb') as fp:
        x_test = pickle.load(fp)

    if problem == 'Cluster':

        if use_only_curated:

            is_curated = train_data['manually_verified'].tolist()
            indx_curated = [i for i,f in enumerate(is_curated) if f == 1]
            x_cur = [x_train[f] for f in indx_curated]

            x_train_2, new_labels_train, mc_new_label_mapping = get_x_and_labels(x_cur,labels_cur,meta_labels_cur,
                                                           cluster = cluster)
        else:
            x_train_2, new_labels_train, mc_new_label_mapping = get_x_and_labels(x_train,labels_all,meta_labels_all,
                                                           cluster = cluster)

        x_test_2, new_labels_test,_ = get_x_and_labels(x_test,labels_test,meta_labels_test,
                                                     cluster = cluster)

        # Load Network
        model = load_network(network,
                             input_shape,
                             len(mc_new_label_mapping)+1,
                             lr,
                             weights = pt_weights,
                             new_head = False,
                             train_only_head = False)

    model.compile(optimizer = Adam(lr), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    model.summary()

    # Make Split
    X_train,X_val,y_train,y_val = train_test_split(x_train_2,new_labels_train,
                                                   test_size=0.1, random_state=7, shuffle= True)

    # Create Generators

    if representation == 'WF':
        train_generator = DataGeneratorWave(X_train,y_train,
                                        batch_size = batch_size,
                                        sr = sr,
                                        file_length = file_length)
        val_generator = DataGeneratorWave(X_val,y_val,
                                          batch_size = batch_size,
                                          sr = sr,
                                          file_length = file_length)
    else:
        train_generator = DataGenerator(X_train,y_train,
                                        batch_size = batch_size,
                                        freq_res = freq_res,
                                        frames = frames)
        val_generator = DataGenerator(X_val,y_val,
                                      batch_size = batch_size,
                                      freq_res = freq_res,
                                      frames = frames)

    # Train Model

    best_filepath = './weights_best.h5'
    checkpoint = ModelCheckpoint(best_filepath,monitor='val_acc',verbose = 1,
                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    history_callback = model.fit_generator(train_generator,epochs = epochs,
                                            validation_data = val_generator,
                                            callbacks = callbacks_list)

    print('\n\nDone Training! Preparing Test\n\n')

    log2 = deepcopy(log)

    log2['acc_history'] = history_callback.history['acc']
    log2['val_acc_history'] = history_callback.history['val_acc']
    log2['loss_history'] = history_callback.history['loss']
    log2['val_loss_history'] = history_callback.history['val_loss']

    model.load_weights(best_filepath)

    if representation == 'WF':
        test_me = create_quick_test_wave(x_test_2,file_length)

    else:
        test_me = create_quick_test_2d(x_test_2,freq_res,frames)

    test_loss,test_acc = model.evaluate(test_me, new_labels_test)
    print("Test Accuracy: {}".format(test_acc))

    y_scores = model.predict(test_me)
    y_hat = np.argmax(y_scores,axis = 1)

    log2['y_scores'] = y_scores
    log2['y_hat'] = y_hat
    log2['test_loss'] = test_loss
    log2['test_acc'] = test_acc

    log['test_loss'] = test_loss
    log['test_acc'] = test_acc
    #print(y_hat)

    print(log)

    version = 0
    while os.path.exists('./outputs/txt_logs/{}[{}].txt'.format(experiment_name,version)):
        version += 1

    with open('./outputs/txt_logs/{}[{}].txt'.format(experiment_name,version), 'w') as f:
        f.write(json.dumps(log, indent=4, separators=(',', ':')))

    with open('./outputs/pickle_logs/{}[{}].p'.format(experiment_name,version),'wb') as fp:
        pickle.dump(log2,fp)

    my_labels = list(mc_new_label_mapping.keys())

    labels = [num_to_label[f] for f in my_labels]
    labels.append('Unknown')

    plot_cm(new_labels_test,y_hat,figsize = (7,7), labels = labels)
    plt.savefig('./outputs/confusion_matrices/{}[{}].eps'.format(experiment_name,version))
    #plt.show()

    model.save_weights('./outputs/weights/{}[{}].h5'.format(experiment_name,version))
    del(model)

if __name__ == '__main__':
    train_model(args)
