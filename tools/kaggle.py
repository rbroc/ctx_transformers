import pandas as pd 
import numpy as np
import tensorflow as tf 
import json
from sklearn.model_selection import train_test_split
from pathlib import Path
import os


def subset_data(data, nr_samples, nr_labels, rand_state=0, **label_kwargs):
    ''' Creates a subset of the dataframe for nr_labels subreddits. Subreddits 
        are chosen randomly. Selects nr_samples examples for each label.
    
    Args:
        data (DataFrame): df including subreddit information
        nr_samples (int): number of samples per subreddit
        nr_labels (int): number of subreddit to randomly sample
    '''
    idx = []
    for s in np.random.choice(data.subreddit.unique(), nr_labels, replace=False):
        idx += list(data[data['subreddit'] == s].sample(nr_samples, random_state=rand_state).index)
    df = data.loc[idx]
    text = list(df.selftext)
    labels = map_labels(df, **label_kwargs)
    return text, labels


def map_labels(data, fname, save=True, labdir='labels'):
    ''' Creates mapping from label strings to numerical indices, and stores in json file '''
    lab_dict = {idx: lab for idx, lab in enumerate(list(data.subreddit.unique()))}
    map_dict = dict(zip(lab_dict.values(), lab_dict.keys()))
    labels = data.subreddit.map(map_dict)
    with open(str(Path(labdir) / fname), 'w') as f:
        json.dump(lab_dict, f)
    return labels


def create_datasets(text, labels, tokenizer, 
                    test_size=.2, val_size=.2,
                    seq_length=512, save=True):
    ''' Tokenizes and creates dataset from list of text and labels ''' 
    train_texts, test_texts, train_labels, test_labels = train_test_split(text, labels, test_size=test_size)
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=val_size)
    # generate encodings
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=20)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=20)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=20)
    # create datasets
    train_ds = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_labels))
    return train_ds, val_ds, test_ds
