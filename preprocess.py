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
from transformers import DistilBertTokenizer
import timeit

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

# Define language detection model anad tokenizer
langdetect = fasttext.load_model('utils/fasttext/lid.176.bin')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


# Define useful functions
def _language_detection(s):
    try:
        s = re.sub('\n', ' ', s)
        return langdetect.predict(' '.join(s.split(' ')[:5]))[0][0].split('__')[2]
    except:
        return 'unk'


def _encode_posts(d):
    idxs = list(np.arange(0, d.shape[0], 100000)) + [d.shape[0]]
    start = timeit.default_timer() 
    timestamp = start
    for i in range(len(idxs) - 1):
        print(f'Timestamp previous step {timestamp - start}')
        print(f'Encoding posts {idxs[i]} to {idxs[i+1]}  \
                out of {d.shape[0]}')
        tokenized = d['selftext'][idxs[i]:idxs[i+1]].apply(lambda x: tokenizer.encode_plus(' '.join(x.split()[:400]), 
                                                                                            truncation=True, 
                                                                                            padding='max_length'))
        if i == 0:
            tokdf = pd.DataFrame(tokenized)
        else:
            tokdf = pd.concat([tokdf, pd.DataFrame(tokenized)], 
                               ignore_index=True)
        timestamp = timeit.default_timer()
    d['input_ids'] = tokdf['selftext'].apply(lambda x: x['input_ids'])
    d['attention_mask'] = tokdf['selftext'].apply(lambda x: x['attention_mask'])
    return d


def _plot_dataset_size(sdict):
    fig, ax = plt.subplots(ncols=3, figsize=(10,4), sharex=True)
    for idx, metric in enumerate(['users', 'posts', 'subreddits']):
        sns.lineplot(x=sdict['names'], y=sdict[metric], ax=ax[idx])
        ax[idx].set_ylabel(metric)
        ax[idx].set_yscale('log')
        ax[idx].set_xticklabels(sdict['names'], rotation=90)
    plt.tight_layout()
    plt.show() # Replace with save


# Preprocessing loop
def preprocess():
    ''' Runs preprocessing on the dataset (filtering by number of users,
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
    df['lang'] = df['selftext'].apply(_language_detection)
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

    # String-ify post text
    df['selftext'] = df['selftext'].astype('str')
    df = _encode_posts(df)
    sizedict = log_size(df, sizedict, 'tokenize', sizelog)

    # Print dataset features
    print(f'There are {df.author.nunique()} users, \
          {df.shape[0]} posts, {df.subreddit.nunique()} subreddits')
    print(f'Min - avg - max posts per user: {df.user_posts_count.min()}, \
                                            {df.user_posts_count.mean()}, \
                                            {df.user_posts_count.max()}')
    print(f'Min - avg - max subreddits per user: {df.user_nr_unique_subreddits.min()}, \
                                                 {df.user_nr_unique_subreddits.mean()}, \
                                                 {df.user_nr_unique_subreddits.max()}')
    print(f'Min - avg - max posts per subreddit: {df.subreddit_posts_count.min()}, \
                                                 {df.subreddit_posts_count.mean()}, \
                                                 {df.subreddit_posts_count.max()}')
    print(f'Min - avg - max users per subreddit: {df.subreddit_nr_unique_users.min()}, \
                                                 {df.subreddit_nr_unique_users.mean()}, \
                                                 {df.subreddit_nr_unique_users.max()}')

    # Save dataset plots
    plot_aggregates(df, bins=[50, 20, 50, 50], vlines=[10, 25, 50, 100], 
                    figsize=(8,6), nrows=2, ncols=2) # Add save option

    # Plot dataset size
    _plot_dataset_size(sizedict)


if __name__ == '__main__':
    preprocess()


