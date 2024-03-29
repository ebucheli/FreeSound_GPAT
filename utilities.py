from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm

import json

def plot_cm(y, y_hat, normalize = True, labels = None, figsize  = (15,15), xrotation = 0):

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
    fig = ax.set_xticklabels(labels, rotation = xrotation)

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

def create_quick_test_2d(x,freq_res,time_res):
    N = len(x)
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

def create_quick_test_wave(x,file_length):

    N = len(x)
    x_2 = np.zeros((N,file_length))

    for i, this_x in enumerate(x):

        this_length = this_x.shape[0]

        if this_length > file_length:
            max_start = this_length - file_length
            start = np.random.randint(0,max_start)
            end = start+file_length

            this_x = this_x[start:end]

        x_2[i] = this_x

    return x_2

def get_all_classes_dict(df):

    classes = np.unique(df['label'])
    n_classes = len(classes)

    num_to_label = dict([[v,k] for v,k in enumerate(classes)])
    label_to_num = dict([[k,v] for v,k in enumerate(classes)])

    return num_to_label, label_to_num, n_classes

def get_classes_to_meta_dict(label_to_num):
    with open('./Clustering_V1_mappings/label_to_meta_v1.json','r') as fp:
        label_to_meta = json.load(fp)

    label_num_to_meta = dict([[label_to_num[f],v] for [f,v] in label_to_meta.items()])

    return label_to_meta, label_num_to_meta

def get_labels(df, label_to_meta, label_to_num):
    #filenames = df['fname'].tolist()
    meta_labels = [label_to_meta[f] for f in df['label']]
    labels = [label_to_num[f] for f in df['label']]
    return meta_labels, labels

def get_x_and_labels(x, labels, meta_labels,cluster = 0):

    indx_mc = [i for i,f in enumerate(meta_labels) if f == cluster]

    x_mc = [x[f] for f in indx_mc]
    labels_mc = [labels[f] for f in indx_mc]

    mc_new_label_mapping = dict([[f,i] for i,f in enumerate(np.unique(labels_mc))])
    new_labels_mc = [mc_new_label_mapping[f] for f in labels_mc]

    indx_unk = [i for i in np.random.randint(0,len(x),len(x_mc))if i not in indx_mc]
    x_unk = [x[f] for f in indx_unk]
    labels_unk = np.ones((len(indx_unk),))*(len(np.unique(new_labels_mc)))

    final_x = x_mc + x_unk
    new_labels_mc.extend(labels_unk)

    return final_x, new_labels_mc, mc_new_label_mapping

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def evaluate_complete_files(x_test,y_test,model,input_shape):

    all_scores = np.zeros((len(x_test),len(np.unique(y_test))))
    scores_list = []

    freq_res,frames = input_shape

    for i,this_x in enumerate(tqdm(x_test)):

        this_len = this_x.shape[1]
        reps = int(np.floor(this_len/frames))

        all_x = np.zeros((reps,freq_res,frames))

        for j in range(reps):

            start = j*frames
            end = start+frames
            my_x = this_x[:,start:end]

            all_x[j] = my_x
        y_hat_all = model.predict(all_x)
        scores_list.append(y_hat_all)
        y_hat_mean = np.mean(y_hat_all,axis = 0)

        all_scores[i] = y_hat_mean

    return all_scores, scores_list
