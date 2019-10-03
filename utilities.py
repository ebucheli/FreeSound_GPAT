from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import json

def plot_cm(y, y_hat, normalize = True, labels = None, figsize  = (15,15)):

    if labels == None:
        labels = np.arange(len(np.unique(y)))

    cm = confusion_matrix(y,y_hat)

    fig, ax = plt.subplots(1,1,figsize = figsize)

    if normalize:
        ax.set_title('Normalized Confusion Matrix')
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        ax.set_title('Confusion Matrix')

    im1 = ax.imshow(cm, cmap = plt.cm.Blues)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im1, cax=cax)

    fig = ax.set(xlabel = 'Predicted', ylabel = 'True')
    fig = ax.set_xticks(np.arange(len(np.unique(y))))
    fig = ax.set_xticklabels(labels)

    fig = ax.set_yticks(np.arange(len(np.unique(y))))
    fig = ax.set_yticklabels(labels)

    fmt = '.2f' if normalize else 'd'

    #fmt = 'd'
    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

def create_quick_test(x,N,freq_res,time_res):
    x_2 = np.zeros((N,freq_res,time_res))

    for i, this_x in enumerate(x):

        this_frames = this_x.shape[1]

        if this_frames > time_res:
            max_start = this_frames - time_res
            start = np.random.randint(0,max_start)
            end = start+time_res

            this_x = this_x[:,start:end]

        x_2[i] = this_x

    return x_2

def get_all_classes_dict(df):

    classes = np.unique(df['label'])
    n_classes = len(classes)

    num_to_label = dict([[v,k] for v,k in enumerate(classes)])
    label_to_num = dict([[k,v] for v,k in enumerate(classes)])

    return num_to_label, label_to_num, n_classes

def get_classes_to_meta_dict(label_to_num):
    with open('./label_to_meta_v1.json','r') as fp:
        label_to_meta = json.load(fp)

    label_num_to_meta = dict([[label_to_num[f],v] for [f,v] in label_to_meta.items()])

    return label_to_meta, label_num_to_meta

def get_labels(df, label_to_meta, label_to_num):
    #filenames = df['fname'].tolist()
    meta_labels = [label_to_meta[f] for f in df['label']]
    labels = [label_to_num[f] for f in df['label']]
    return meta_labels, labels
