import ray
import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from dask import dataframe as dd
from dask.distributed import Client
import warnings
import dask.array as da
from tqdm import tqdm
from dask_ml.metrics import mean_squared_error
warnings.filterwarnings('ignore')
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz
import time
import os
import ray
import random


@ray.remote
def transform(part, info_name, model, n_clusters, components, p):
    current = part[info_name].compute().values.reshape(-1, 1)

    # means and stds of the modes are obtained from the corresponding fitted bgm model
    means = model.means_.reshape((1, n_clusters))
    stds = np.sqrt(model.covariances_).reshape((1, n_clusters))

    # values are then normalized and stored for all modes
    features = np.empty(shape=(len(current), n_clusters))
    # note 4 is a multiplier to ensure values lie between -1 to 1 but this is not always guaranteed
    #print(current, means, stds)
    features = (current - means) / (4 * stds)

    # number of distict modes
    n_opts = sum(components)

    # storing the mode for each data point by sampling from the probability mass distribution across all modes based on fitted bgm model
    opt_sel = np.zeros(len(current), dtype='int')

    probs = model.predict_proba(current)

    #print('prediction complete')
    probs = probs[:, components]
    probs = probs + 1e-6
    probs = probs / np.broadcast_to(probs.sum(axis=1).reshape((len(probs), 1)), probs.shape)
    opt_sel = (probs.cumsum(axis=1) <= np.broadcast_to(np.random.random(len(probs)).reshape((len(probs), 1)), probs.shape)).sum(axis=1)
    # creating a one-hot-encoding for the corresponding selected modes
    #print(opt_sel, 'opt_sel')
    probs_onehot = np.eye(probs.shape[1])[opt_sel]
    # obtaining the normalized values based on the appropriately selected mode and clipping to ensure values are within (-1,1)
    features = features[:, components]
    features = (features * probs_onehot).sum(axis=1).reshape([-1, 1])
    features = np.clip(features, -.99, .99)

     # re-ordering the one-hot-encoding of modes in descending order as per their frequency of being selected
    re_ordered_phot = np.zeros_like(probs_onehot)
    col_sums = probs_onehot.sum(axis=0)
    #print(col_sums, 'col_sums')
    n = probs_onehot.shape[1]
    largest_indices = np.argsort(-1*col_sums)[:n]
    #print(largest_indices)
    for id,val in enumerate(largest_indices):
        re_ordered_phot[:,id] = probs_onehot[:,val]
    

    # storing transformed numeric column represented as normalized values and corresponding modes
    save_npz(f'../data/transformed/transformed_{p}.npz', csr_matrix(np.concatenate([features, re_ordered_phot], axis=1)))

    return largest_indices