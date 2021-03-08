import pandas as pd
import numpy as np
import wget
import os
from utils import (read_files, compute_aggregates, 
                   plot_aggregates, update_aggregates, 
                   log_size, plot_size_log)
import fasttext
from pathlib import Path
import argparse


# Set up command line args and parser
parser = argparse.ArgumentParser()
parser.add_argument('--min-posts', type=int, default=5,
                    help='Minimum number of posts per user')
parser.add_argument('--min-subreddits', type=int, default=5,
                    help='Minimum number of subreddits per user')


# Define paths
DATA_PATH = Path('..') / 'data'
RAW_PATH = DATA_PATH / 'raw'
RAW_PATH.mkdir(exist_ok=True, parents=True)
PROCESSED_PATH = DATA_PATH / 'filtered'
PROCESSED_PATH.mkdir(exist_ok=True)
LOG_PATH = PROCESSED_PATH / 'log.json'
FIG_PATH = DATA_PATH / 'figures'
FIG_PATH.mkdir(exist_ok=True)
TMP_PATH = DATA_PATH / 'tmp'
TMP_PATH.mkdir(exist_ok=True)
FASTTEXT_FILE = TMP_PATH / 'lid.176.bin'

# Define language detection model and tokenizer
FASTTEXT_URL = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin'
wget.download(FASTTEXT_URL, out=str(FASTTEXT_FILE))
langdetect = fasttext.load_model(str(FASTTEXT_FILE))


# Define useful functions
def _language_detection(s):
    try:
        s = re.sub('\n', ' ', s)
        return langdetect.predict(s)[0][0].split('__')[2]
    except:
        return 'unk'


def preprocess(min_posts=5, min_subreddits=5):
    ''' Runs preprocessing on the dataset (filtering duplicates,
        filtering by minimum number of posts per user, removing 
        non-English posts) 
    Args:
        min_posts (int): minimum number of posts per user
        min_subreddits (int): mininum number of subreddits to 
            which the user has to have contributed    
    '''
    print('Reading files...')

    # Load and merge
    df = read_files(RAW_PATH)
    df = df[~df['subreddit'].isnull()]
    df = df.drop_duplicates(subset=['author', 
                                    'selftext'])
    ldict = log_size(df, 'filter_duplicates', 
                     save_file=LOG_PATH)
    print(f'\nPosts at load: {ldict["posts"][-1]}')

    # Filter users by min number of posts and subreddits
    print('Filtering by post/subreddit threshold...')
    df = compute_aggregates(df, 
                            group_by='author', 
                            agg_fn='count', 
                            colnames=['author', 
                                      'n_user_posts'])
    df = compute_aggregates(df, group_by='author', 
                            agg_fn=lambda x: x.nunique(), 
                            colnames=['author', 
                                      'n_user_subreddits'], 
                            target='subreddit')
    df = df[(df['n_user_posts'] >= min_posts) & \
            (df['n_user_subreddits'] >= min_subreddits)]
    ldict = log_size(df, ldict, 'filter_users', LOG_PATH)
    print(f'\tPosts at filter_users: {ldict["posts"][-1]}')

    # Remove non-English posts
    print('Removing non-English posts...')
    df['lang'] = df['selftext'].apply(_language_detection)
    df = df[df['lang'] == 'en']
    ldict = log_size(df, ldict, 'filter_lang', LOG_PATH)
    print(f'\tPosts at filter_lang: {ldict["posts"][-1]}')
    os.remove(FASTTEXT_FILE)
    os.rmdir(TMP_PATH)
    
    # Log and save (split if files with 1000 users each)
    print('Saving filtered dataset...')
    df = update_aggregates(df)
    df['user_id'] = df['author'].map(dict(zip(df.author.unique(),
                                              range(df.author.nunique()))))
    idxs = np.arange(0, ldict['users'][-1], 1000)
    idxs = idxs.append(ldict['users'][-1])
    for idx in idxs[:-1]:
        print(f'\tSaving {idx} of {idxs[-1]}')
        subdf = df[df['author'].isin(df.author.unique()[idx:idx+1])]
        fname = PROCESSED_PATH / f'filtered_{idx+1}.txt'
        subdf.to_csv(fname, sep='\t', index=False, compression='gzip')
    del subdf

    # Print dataset features
    print(f'''\nThere are {ldict["users"][-1]} users, 
                          {ldict["posts"][-1]} posts,
                          {ldict["subreddits"][-1]} subreddits''')
    pmetrics = df.n_user_posts.agg(['min', 'max', 'mean']).tolist()
    smetrics = df.n_user_subeddits.agg(['min', 'max', 'mean']).tolist()
    print(f'Min, avg, max posts per user: {pmetrics}')
    print(f'Min, avg, max subreddits per user: {smetrics}')

    # Save aggregates and size plots
    fname_agg = str(FIG_PATH / 'aggregates.png')
    plot_aggregates(df, save_file=fname_agg)
    fname_size = str(FIG_PATH / 'preprocessing_log.png')
    plot_size_log(ldict, save_file=fname_size)


if __name__ == '__main__':
    args = parser.parse_args()
    preprocess(args.min_posts, args.min_subreddits)


