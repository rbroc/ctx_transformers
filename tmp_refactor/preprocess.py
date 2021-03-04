import json
import os
import pandas as pd
import numpy as np
import random
import re
import itertools
from tools.preprocess import (merge_csv, add_aggregate_metrics, 
                              plot_aggregates, update_aggregates, 
                              log_size, encode_posts, plot_size_curve)
import fasttext
from pathlib import Path
from transformers import DistilBertTokenizer
import timeit

# define paths
RAW_PATH = Path('raw') / 'pushshift'
PROCESSED_PATH = Path('processed') / 'pushshift'
FIG_PATH = Path('figures')
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

# Define language detection model and tokenizer
langdetect = fasttext.load_model('utils/fasttext/lid.176.bin')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Define useful functions
def _language_detection(s):
    try:
        s = re.sub('\n', ' ', s)
        return langdetect.predict(' '.join(s.split(' ')[:5]))[0][0].split('__')[2]
    except:
        return 'unk'


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
    print('Subsetting top 500 subreddits...')
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
    sizedict = log_size(df, sizedict, '500sr_drop_duplicates', sizelog)

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
    df.to_csv(str(PROCESSED_PATH / '500sr.txt'), sep='\t', index=False)
    sizedict = log_size(df, sizedict, '500sr_user_filter', sizelog)

    # String-ify post text
    df['selftext'] = df['selftext'].astype('str')
    df = encode_posts(df, tokenizer)
    sizedict = log_size(df, sizedict, 'tokenize', sizelog)
    df.to_csv('processed/pushshift/500sr_encoded.txt', sep='\t', index=False)

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
                    figsize=(8,6), nrows=2, ncols=2, save=True,
                    fname=str(FIG_PATH/'pushshift_descriptives.png'))

    # Plot dataset size
    plot_size_curve(sizedict, save=True, 
                    fname=str(FIG_PATH/'pushshift_size.png'))


if __name__ == '__main__':
    preprocess()


