import pandas as pd
import numpy as np
from reddit.utils import (read_files, compute_aggregates, 
                          plot_aggregates, update_aggregates, 
                          log_size, plot_size_log)
import fasttext
from pathlib import Path
from transformers import DistilBertTokenizer

### See to dos in notes ### 

# Define paths
RAW_PATH = Path('..') / 'data' / 'raw'
RAW_PATH.mkdir(exist_ok=True, parents=True)
PROCESSED_PATH = Path('..') / 'data' / 'processed'
PROCESSED_PATH.mkdir(exist_ok=True)
FIG_PATH = Path('..') / 'data' / 'figures'
FIG_PATH.mkdir(exist_ok=True)
LOG_PATH = Path('..') / 'data' / 'logs'
LOG_PATH.mkdir(exist_ok=True)

# define params for filtering
MIN_POSTS = 5
MIN_SUBREDDITS = 5

# Define language detection model and tokenizer
langdetect = fasttext.load_model('utils/fasttext/lid.176.bin')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Define useful functions
def _language_detection(s):
    try:
        s = re.sub('\n', ' ', s)
        return langdetect.predict(s)[0][0].split('__')[2]
    except:
        return 'unk'

# Preprocessing loop
def preprocess():
    ''' Runs preprocessing on the dataset (filtering by number of users,
        subreddits, language, duplicates, etc...)'''
    print('Reading files...')

    # Load and merge
    df = read_files(PROCESSED_PATH)
    df = df[~df['subreddit'].isnull()]
    df = df.drop_duplicates(subset=['author', 
                                    'selftext'])
    ldict = log_size(df, 'filter_duplicates', 
                     save_file=LOG_PATH)
    print(f'\tPosts at load: {ldict["posts"][-1]}')

    # Filter users for min number of posts and subreddits
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
    df = df[(df['n_user_posts'] >= MIN_POSTS) & \
            (df['n_user_subreddits'] >= MIN_SUBREDDITS)]
    ldict = log_size(df, ldict, 'filter_users', LOG_PATH)
    print(f'\tPosts at filter_users: {ldict["posts"][-1]}')

    # Remove non-English posts
    print('Removing non-English posts...')
    df['lang'] = df['selftext'].apply(_language_detection)
    df = df[df['lang'] == 'en']
    ldict = log_size(df, ldict, 'filter_lang', LOG_PATH)
    df = update_aggregates(df)
    print(f'\tPosts at filter_lang: {ldict["posts"][-1]}')

    # Log and save
    print('Saving filtered dataset...')
    fname = PROCESSED_PATH / 'filtered.txt'
    df.to_csv(fname, sep='\t', index=False, compression='gzip')


    # Print dataset features
    print(f'''There are {ldict["users"][-1]} users, 
                        {ldict["posts"][-1]} posts,
                        {ldict["subreddits"][-1]} subreddits
         ''')
    pmin, pmean, pmax = df.n_user_posts.agg(['min', 'max', 'mean'])
    smin, smean, smax = df.n_user_subeddits.agg(['min', 'max', 'mean'])
    print(f'Min, avg, max posts per user: {pmin}, {pmean}, {pmax}')
    print(f'Min, avg, max subreddits per user: {smin}, {smean}, {smax)}')

    # Save dataset plots
    fname_agg = str(FIG_PATH / 'dataset_metrics.png')
    plot_aggregates(df, save_file=fname_agg)

    # Plot dataset size
    fname_size = str(FIG_PATH / 'dataset_size_log.png')
    plot_size_log(ldict, save_file=fname_size)


if __name__ == '__main__':
    preprocess()


