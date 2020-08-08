#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Usage:
    $python3 lightfm_recsys.py interactions_data_file_path subsampling_datasize
"""

import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from IPython.display import display_html
import warnings
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
from lightfm import LightFM
import time

# update the working directory to the root of the project
# os.chdir('..')
warnings.filterwarnings("ignore")


def load_data(file_name):
    return pd.read_csv(file_name)


def clean_data(data):
    interactions_selected = data.loc[data['is_read'] == 1, ['user_id', 'book_id', 'rating']]
    interactions_selected = interactions_selected[
        interactions_selected['user_id'].isin(list(interactions_selected['user_id'].unique()))]
    return interactions_selected


def down_sampling(data, k):
    return data.sample(k)


def csr(cleaned_interactions_data):
    user_book_interaction = pd.pivot_table(cleaned_interactions_data, index='user_id', columns='book_id',
                                           values='rating')
    # fill missing values with 0
    user_book_interaction = user_book_interaction.fillna(0)
    user_book_interaction_csr = csr_matrix(user_book_interaction.values)
    return user_book_interaction_csr


def train_val_split(csr_mat):
    train, val = random_train_test_split(csr_mat, test_percentage=0.2)
    return (train, val)


def fit(train):
    model = LightFM(loss='warp',
                    random_state=2016,
                    learning_rate=0.01,
                    no_components=30,
                    user_alpha=0.000005,
                    item_alpha=0.000005)
    start_time = time.time()
    model = model.fit(train,
                      epochs=100,
                      num_threads=16, verbose=False)
    end_time = time.time()
    print("Training time is:", end_time - start_time)
    return model


def eval(model, train, val):
    # auc
    print("Train auc: %.2f" % auc_score(model, train).mean())
    print("Val auc: %.2f" % auc_score(model, val).mean())
    # precision_at_k
    print("Train precision: %.2f" % precision_at_k(model, train, k=5).mean())
    print("Val precision: %.2f" % precision_at_k(model, val, k=5).mean())
    # recall_at_k
    print("Train recall: %.2f" % precision_at_k(model, train, k=5).mean())
    print("Val recall: %.2f" % precision_at_k(model, val, k=5).mean())


if __name__ == '__main__':
    # Get the parameters from the command line
    file = sys.argv[1]
    datasize = int(sys.argv[2])

    # Call our main routine
    csr_matrix_ = down_sampling(clean_data(load_data(file)), k=datasize)
    train, val = train_val_split(csr_matrix_)
    model = fit(train)
    eval(model, train, val)
