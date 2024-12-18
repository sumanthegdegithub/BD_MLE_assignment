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
from scipy.sparse import csr_matrix, save_npz, load_npz, vstack
import time
import os
import random
from pathlib import Path
import sys
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
import ray
import polars as pl
from numba import njit

from app.logger import logger
import yaml
import threading
import queue

with open(f"{str(root)}\\config.yml", "r") as f:
    config = yaml.safe_load(f)

context = ray.init()
print(context.dashboard_url)

@njit
def get_features(current, means, stds):
    return (current - means) / (4 * stds), np.zeros(len(current), dtype='int')

@ray.remote
class BackgroundActor:
    def ray_save(self, features, re_ordered_phot, p):
        pl.from_numpy(np.concatenate([features, re_ordered_phot], axis=1)).write_parquet(f'{str(root)}\\data\\transformed\\part.{p}.parquet')
        logger.info(f'saved {p}')


class DataTransformer():

    """
    Transformer class responsible for processing data to train the CTABGANSynthesizer model

    Variables:
    1) train_data -> input dataframe
    2) categorical_list -> list of categorical columns
    3) mixed_dict -> dictionary of mixed columns
    4) n_clusters -> number of modes to fit bayesian gaussian mixture (bgm) model
    5) eps -> threshold for ignoring less prominent modes in the mixture model
    6) ordering -> stores original ordering for modes of numeric columns
    7) output_info -> stores dimension and output activations of columns (i.e., tanh for numeric, softmax for categorical)
    8) output_dim -> stores the final column width of the transformed data
    9) components -> stores the valid modes used by numeric columns
    10) filter_arr -> stores valid indices of continuous component in mixed columns
    11) meta -> stores column information corresponding to different data types i.e., categorical/mixed/numerical


    Methods:
    1) __init__() -> initializes transformer object and computes meta information of columns
    2) get_metadata() -> builds an inventory of individual columns and stores their relevant properties
    3) fit() -> fits the required bgm models to process the input data
    4) transform() -> executes the transformation required to train the model
    5) inverse_transform() -> executes the reverse transformation on data generated from the model

    """

    def __init__(self, train_data=pd.DataFrame, categorical_list=[], mixed_dict={}, n_clusters=config['n_clusters'], eps=config['eps']):

        self.meta = None
        self.train_data = train_data
        self.categorical_columns= categorical_list
        self.mixed_columns= mixed_dict
        self.n_clusters = n_clusters
        self.eps = eps
        self.ordering = []
        self.output_info = []
        self.output_dim = 0
        self.components = []
        self.filter_arr = []
        self.meta = self.get_metadata()

    def get_metadata(self):

        try:
            meta = []
            for index in range(self.train_data.shape[1]):
                column = self.train_data.iloc[:,index]
                if index in self.categorical_columns:
                    mapper = column.value_counts().index.tolist()
                    meta.append({
                            "name": index,
                            "type": "categorical",
                            "size": len(mapper),
                            "i2s": mapper
                    })
                elif index in self.mixed_columns.keys():
                    meta.append({
                        "name": index,
                        "type": "mixed",
                        "min": column.min(),
                        "max": column.max(),
                        "modal": self.mixed_columns[index]
                    })
                else:
                    meta.append({
                        "name": self.train_data.columns[index],
                        "type": "continuous",
                        "min": float(column.min().compute()),
                        "max": float(column.max().compute()),
                    })

            return meta
    
        except Exception as e:
            exc_type, exc_value, exc_tb = sys.exc_info()
            fname = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            logger.info(f'Error: {exc_value} occured in file {fname} at line {line_number}')
            return []
        

    def fit(self):
        try:
            data = self.train_data.values
            # stores the corresponding bgm models for processing numeric data
            model = []
            # iterating through column information
            for id_, info in enumerate(self.meta):
                if info['type'] == "continuous":
                    # fitting bgm model
                    logger.info('model training has started')
                    #weight_concentration_prior = 
                    gm = BayesianGaussianMixture(
                        n_components = self.n_clusters,
                        weight_concentration_prior_type='dirichlet_process',
                        weight_concentration_prior=0.001, # lower values result in lesser modes being active
                        max_iter=10,n_init=1, random_state=42, warm_start = True)
                    
                    means_initiation = 0
                    records_trained_on = 0
                    parts = list(data.partitions)
                    epochs = 10
                    for p in range(len(parts) * epochs):
                        r = random.randint(0, len(parts)-1)
                        part = parts[r].compute()        
                        gm.fit(part.reshape([-1, 1]))
                        records_trained_on+=part.shape[0]
                        if (p)%20 == 19:
                            if means_initiation == 0:
                                means_initiation=1
                                means_prior =gm.means_
                                weights_prior = gm.weights_
                                covariances_ = gm.covariances_
                            else:
                                if (np.abs(covariances_ - gm.covariances_).sum() + np.abs(means_prior - gm.means_).sum() + np.abs(weights_prior - gm.weights_).sum()) < 1e-2:
                                    logger.info(f'early stopping the training, records_trained_on {records_trained_on}')
                                    break
                                
                                #print(np.abs(covariances_ - gm.covariances_).sum(), np.abs(means_prior - gm.means_).sum(), np.abs(weights_prior - gm.weights_).sum())
                                means_prior =gm.means_
                                weights_prior =gm.weights_
                                covariances_ = gm.covariances_

                            logger.info(f'training in progress, records_trained_on {records_trained_on}')

                    logger.info('model training completed')
                    model.append(gm)
                    # keeping only relevant modes that have higher weight than eps and are used to fit the data
                    old_comp = gm.weights_ > self.eps
                    mode_freq = []
                    for p in range(min(10, len(parts))): 
                        r = random.randint(0, len(parts)-1)
                        part = parts[r].compute()   
                        mode_freq += list(pd.Series(gm.predict(part.reshape([-1, 1]))).value_counts().keys())
                    mode_freq = set(mode_freq)
                    comp = []
                    for i in range(self.n_clusters):
                        if (i in (mode_freq)) & old_comp[i]:
                            comp.append(True)
                        else:
                            comp.append(False)
                    self.components.append(comp)
                    self.output_info += [(1, 'tanh'), (np.sum(comp), 'softmax')]
                    self.output_dim += 1 + np.sum(comp)

            self.model = model
        except Exception as e:
            exc_type, exc_value, exc_tb = sys.exc_info()
            fname = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            logger.info(f'Error: {exc_value} occured in file {fname} at line {line_number}')

    def transform(self, data):
        try:
            # stores the transformed values
            values = []
            # iterating through column information
            logger.info('transformation started')
            actor = BackgroundActor.remote()
            for id_, info in enumerate(self.meta):
                data_queue_part1 = queue.Queue()
                data_queue_part2 = queue.Queue()
                
                if info['type'] == "continuous":
                    means = self.model[id_].means_.reshape((1, self.n_clusters))
                    stds = np.sqrt(self.model[id_].covariances_).reshape((1, self.n_clusters))
                    
                    n_opts = sum(self.components[id_])
                    values = []
                    memory = pd.DataFrame()

                    def part_one(parts):
                        parts = list(data.partitions)
                        for p in tqdm(range(len(parts))):
                            data_queue_part1.put((parts[p].compute(), p))
                        data_queue_part1.put(None)

                    def part_two(n_opts):
                        memory = pd.DataFrame()
                        
                        arange = np.arange(n_opts)

                        while True:
                            data = data_queue_part1.get()

                            if data is None:  # Stop signal
                                data_queue_part2.put(None) 
                                break
                            t = time.time()
                            part, p = data
                            part = part[[info['name']]]
                            if memory.shape[0] > 0:
                                part_not_in_memory = part[~part[info['name']].isin(memory[info['name']])]
                            else:
                                part_not_in_memory = part.copy()
                            
                            if len(part_not_in_memory) > 0:
                                if memory.shape[0]> 0:
                                    current = part_not_in_memory.drop_duplicates(info['name'])[[info['name']]].values
                                    probs = self.model[id_].predict_proba(current.reshape([-1, 1]))
                                    probs = probs[:, self.components[id_]] + 1e-6
                                    part_not_in_memory[arange] = probs
                                    part = part.merge(memory, on=info['name'], how='left')
                                    part = part.set_index(info['name']).combine_first(part_not_in_memory.set_index(info['name'])).reset_index()
                                    #part.loc[~part[info['name']].isin(memory[info['name']]), np.arange(probs.shape[1])] = probs
                                    probs = part[arange].values
                                    if memory.shape[0] < 1e5:
                                        memory = pd.concat([memory, part_not_in_memory.drop_duplicates(info['name'])])
                                else:
                                    current = part_not_in_memory[[info['name']]].values
                                    probs = self.model[id_].predict_proba(current.reshape([-1, 1]))
                                    probs = probs[:, self.components[id_]] + 1e-6
                                    part_not_in_memory[arange] = probs
                                    memory = part_not_in_memory.drop_duplicates(info['name']).copy()
                            else:
                                part = part.merge(memory, on=info['name'], how='left')
                                probs = part.drop(info['name'], axis=1).values

                            current = part[[info['name']]].values

                            data_queue_part2.put((current, probs, p)) 

                        

                    def part_three(means, stds, actor):
                            
                            while True:

                                data = data_queue_part2.get()

                                if data is None:  # Stop signal
                                    break

                                current, probs, p = data
                                
                                #print(current, means, stds)
                                features, opt_sel = get_features(current, means, stds)

                                #probs = probs[:, self.components[id_]] + 1e-6
                                probs = probs / np.broadcast_to(probs.sum(axis=1).reshape((len(probs), 1)), probs.shape)
                                opt_sel = (probs.cumsum(axis=1) <= np.broadcast_to(np.random.random(len(probs)).reshape((len(probs), 1)), probs.shape)).sum(axis=1)

                                # creating a one-hot-encoding for the corresponding selected modes

                                probs_onehot = np.eye(probs.shape[1])[opt_sel]
                                # obtaining the normalized values based on the appropriately selected mode and clipping to ensure values are within (-1,1)
                                features = features[:, self.components[id_]]
                                features = (features * probs_onehot).sum(axis=1).reshape([-1, 1])
                                
                                features = np.clip(features, -.99, .99)

                                # re-ordering the one-hot-encoding of modes in descending order as per their frequency of being selected
                                #re_ordered_phot = np.zeros_like(probs_onehot)
                                col_sums = probs_onehot.sum(axis=0)
                                #print(col_sums, 'col_sums')
                                n = probs_onehot.shape[1]
                                largest_indices = np.argsort(-1*col_sums)[:n]

                                #print(largest_indices)
                                re_ordered_phot = probs_onehot[:, largest_indices]
                                
                                # storing the original ordering for invoking inverse transform
                                self.ordering.append(largest_indices)

                                # storing transformed numeric column represented as normalized values and corresponding modes
                                actor.ray_save.remote(features, re_ordered_phot, p)

                    thread1 = threading.Thread(target=part_one,args=(data,))
                    thread2 = threading.Thread(target=part_two,args=(n_opts,))
                    thread3 = threading.Thread(target=part_three,args=(means, stds, actor))

                    thread1.start()
                    thread2.start()
                    thread3.start()

                    # Wait for threads to complete
                    thread1.join()
                    thread2.join()
                    thread3.join()


            logger.info('transformation complete')
        except Exception as e:
            exc_type, exc_value, exc_tb = sys.exc_info()
            fname = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            logger.info(f'Error: {exc_value} occured in file {fname} at line {line_number}')
        

    def inverse_transform(self):
        try:

            logger.info('inverse transformation started')
            data_loc = config['transformed_file_loc']
            trans_formed_files = pd.Series(os.listdir(f'{str(root)}\\data\\transformed\\'))
            trans_formed_files_dict = dict(zip(trans_formed_files.str.split('.').str[1].astype(int), trans_formed_files))
            
            for i in tqdm(trans_formed_files_dict.keys()):
                data = pl.read_parquet(f'{str(root)}\\data\\transformed\\' + trans_formed_files_dict[i]).to_numpy()

                # used to iterate through the columns of the raw generated data
                st = 0

                # iterating through original column information
                for id_, info in enumerate(self.meta):
                    if info['type'] == "continuous":

                        # obtaining the generated normalized values and clipping for stability
                        u = data[:, st]
                        u = np.clip(u, -1, 1)

                        # obtaining the one-hot-encoding of the modes representing the normalized values
                        v = data[:, st + 1:st + 1 + np.sum(self.components[id_])]
                        
                        # re-ordering the modes as per their original ordering
                        order = self.ordering[id_]
                        
                        v_re_ordered = np.empty_like(v)
                        v_re_ordered[:, order] = v[:, np.arange(len(order))]

                        v = v_re_ordered

                        # ensuring un-used modes are represented with -100 such that they can be ignored when computing argmax
                        v_t = np.ones((data.shape[0], self.n_clusters)) * -100
                        v_t[:, self.components[id_]] = v
                        v = v_t

                        # obtaining approriate means and stds as per the appropriately selected mode for each data point based on fitted bgm model
                        means = self.model[id_].means_.reshape([-1])
                        stds = np.sqrt(self.model[id_].covariances_).reshape([-1])
                        p_argmax = np.argmax(v, axis=1)
                        std_t = stds[p_argmax]
                        mean_t = means[p_argmax]

                        # executing the inverse transformation
                        tmp = u * 4 * std_t + mean_t

                        pd.DataFrame({
                            info['name']: tmp,
                        }).to_parquet(f'{str(root)}\\data\\inverse_transformed\\part.{i}.parquet')

            logger.info('inverse transformation completed')

        except Exception as e:
            exc_type, exc_value, exc_tb = sys.exc_info()
            fname = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            logger.info(f'Error: {exc_value} occured in file {fname} at line {line_number}')