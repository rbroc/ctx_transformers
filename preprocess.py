import json
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import seaborn as sns
from matplotlib import pyplot as plt
import re
import itertools
from tools.preprocess import (merge_csv, add_aggregate_metrics, 
                              plot_aggregates, update_aggregates, 
                              log_size)
from tools.datasets import (save_tfrecord_nn1, load_tfrecord_nn1)
import fasttext
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path


# define paths
RAW_PATH = Path('raw') / 'pushshift'
PROCESSED_PATH = Path('processed') / 'pushshift'
RAW_PATH.mkdir(exist_ok=True)
PROCESSED_PATH.mkdir(exist_ok=True)


# define params for filtering
MIN_POSTS_PER_USER = 10
MIN_UNIQUE_SUBREDDITS_PER_USER = 3
MIN_POSTS_PER_SUBREDDIT = 10
MIN_UNIQUE_USERS_PER_SUBREDDIT = 10


# Define log parameters
sizedict = {'names':[], 'users':[], 'posts':[], 'subreddits':[]}
sizelog = str(PROCESSED_PATH / 'size_log.json')
sizedict = json.load(open(sizelog))


# Define language detection model
langdetect = fasttext.load_model('utils/fasttext/lid.176.bin')


# Define useful functions
def language_detection(s):
    try:
        s = re.sub('\n', ' ', s)
        return langdetect.predict(' '.join(s.split(' ')[:5]))[0][0].split('__')[2]
    except:
        return 'unk'


def preprocess():
    ''' Does first preprocessing on the dataset (filtering by number of users,
        subreddits, language, duplicates, etc...)'''

    print('Reading files...')
    # Load and merge
    df = merge_csv(PROCESSED_PATH)
    df = df[~df['subreddit'].isnull()]
    sizedict = log_size(df, sizedict, 'no_filter', sizelog)

    # Filter users for min number of posts and subreddits
    print('Filtering by user characteristics...')
    df = add_aggregate_metrics(df, 'author', 'count', ['author', 'user_posts_count'])
    df = add_aggregate_metrics(df, 'author', lambda x: x.nunique(), 
                            ['author', 'user_nr_unique_subreddits'], 
                            agg_on='subreddit')
    df = df[(df['user_posts_count'] >= MIN_POSTS_PER_USER) & \
            (df['user_nr_unique_subreddits'] >= MIN_UNIQUE_SUBREDDITS_PER_USER)]
    sizedict = log_size(df, sizedict, 'user_filter', sizelog)

    # Filter subreddits for minimum number of posts and users
    print('Filtering by subreddit characteristics...')
    df = add_aggregate_metrics(df, 'subreddit', 'count', 
                            ['subreddit', 'subreddit_posts_count'])
    df = add_aggregate_metrics(df, 'subreddit', lambda x: x.nunique(), 
                            ['subreddit', 'subreddit_nr_unique_users'], agg_on='author')
    df = df[(df['subreddit_posts_count'] >= MIN_POSTS_PER_SUBREDDIT) & \
            (df['subreddit_nr_unique_users'] >= MIN_UNIQUE_USERS_PER_SUBREDDIT)]
    sizedict = log_size(df, sizedict, 'subreddit_filter', sizelog)

    # Filter by language 
    print('Removing non-English posts')
    df['lang'] = df['selftext'].apply(language_detection)
    df = df[df['lang'] == 'en']
    sizedict = log_size(df, sizedict, 'lang_filter', sizelog)

    # Log and save
    print('Saving filtered dataset...')
    df = update_aggregates(df)
    fname = PROCESSED_PATH / 'filtered.txt'
    df.to_csv(fname, sep='\t', index=False)

    # Subset subreddits
    print('Subsetting for top 500 subreddits...')
    srdict = {}
    for sr in df.subreddit.unique():
        srdict[sr] = df[df['subreddit'] == sr]['subreddit_posts_count'].iloc[0]
    srdict = [(k,v) for k, v in sorted(srdict.items(), reverse=True, key=lambda item: item[1])]
    srdict = srdict[:500]
    srdict = [s[0] for s in srdict]
    df = df[df['subreddit'].isin(srdict)]
    df = update_aggregates(df)
    sizedict = log_size(df, sizedict, '500sr_subset', sizelog)

    # Drop duplicates
    print('Dropping duplicates')
    df = df.drop_duplicates('selftext')
    df = update_aggregates(df)
    sizedict = log_size(df, sizedict, '5000sr_drop_duplicates', sizelog)

    # Check number of users per subreddit
    df = df[(df['user_posts_count'] >= MIN_POSTS_PER_USER) & \
            (df['user_nr_unique_subreddits'] >= MIN_UNIQUE_SUBREDDITS_PER_USER)]
    df = update_aggregates(df)
    assert all(df['subreddit_posts_count'] >= MIN_POSTS_PER_SUBREDDIT), \
        'Too few posts peer subreddit'
    assert all(df['subreddit_nr_unique_users'] >= MIN_UNIQUE_USERS_PER_SUBREDDIT), \
                'Too few users per subreddit!'

    # Save and log
    print('Saving filtered file...')
    df.to_csv(str(PROCESSED_PATH / 'dataset_500sr.txt'), sep='\t', index=False)
    sizedict = log_size(df, sizedict, '500sr_user_filter', sizelog)


if __name__ == '__main__':
    preprocess()





