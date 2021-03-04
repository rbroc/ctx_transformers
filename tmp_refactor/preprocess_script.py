import pandas as pd
import numpy as np
from tools.preprocess import (merge_csv, add_aggregate_metrics, 
                              plot_aggregates, update_aggregates, 
                              log_size, encode_posts, plot_size_curve)
import fasttext
from pathlib import Path
from transformers import DistilBertTokenizer

### See to dos in notes ### 

# Define paths
RAW_PATH = Path('raw') / 'pushshift'
PROCESSED_PATH = Path('processed') / 'pushshift'
FIG_PATH = Path('figures')
RAW_PATH.mkdir(exist_ok=True)
PROCESSED_PATH.mkdir(exist_ok=True)
SIZELOG_PATH = Path('..') / 'data' / 'logs'

# define params for filtering
MIN_USER_POSTS = 5
MIN_USER_SUBREDDITS = 5

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
    df = merge_csv(PROCESSED_PATH)
    df = df[~df['subreddit'].isnull()]
    sizedict = log_size(df, 'raw', SIZELOG_PATH)

    # REMOVE DUPLICATES HERE?

    # Filter users for min number of posts and subreddits
    print('Filtering by user characteristics...')
    df = add_aggregate_metrics(df, group_by='author', agg_fn='count', 
                               colnames=['author', 'n_user_posts'])
    df = add_aggregate_metrics(df, group_by='author', agg_fn=lambda x: x.nunique(), 
                               colnames=['author', 'n_user_subreddits'], 
                               target='subreddit')
    df = df[(df['n_user_posts'] >= MIN_USER_POSTS) & \
            (df['n_user_subreddits'] >= MIN_USER_SUBREDDITS)]
    sizedict = log_size(df, sizedict, 'filter_users', SIZELOG_PATH)

    # Filter by language 
    print('Removing non-English posts...')
    df['lang'] = df['selftext'].apply(_language_detection)
    df = df[df['lang'] == 'en']
    sizedict = log_size(df, sizedict, 'filter_language', SIZELOG_PATH)

    # Drop duplicates
    print('Dropping duplicates...')
    df = df.drop_duplicates('selftext')
    df = update_aggregates(df)
    sizedict = log_size(df, sizedict, 'filter_duplicates', SIZELOG_PATH) 

    # Log and save
    print('Saving filtered dataset...')
    df = update_aggregates(df)
    fname = PROCESSED_PATH / 'filtered.txt'
    df.to_csv(fname, sep='\t', index=False, compression='gzip')

    # Tokenize posts
    df['selftext'] = df['selftext'].astype('str')
    df = encode_posts(df, tokenizer)
    sizedict = log_size(df, sizedict, 'tokenize', SIZELOG_PATH)
    df.to_csv('processed/pushshift/encoded.txt', sep='\t', index=False, compression='gzip') # Define name above

    # Print dataset features
    print(f'''There are {sizedict["users"][-1]} users, 
              {sizedict["posts"][-1]} posts,
              {sizedict["subreddits"][-1]} subreddits''')
    print(f'Min, avg, max posts per user: {df.n_user_posts.min()}, \
                                          {df.n_user_posts.mean()}, \
                                          {df.n_user_posts.max()}')
    print(f'Min, avg, max subreddits per user: {df.n_user_subreddits.min()}, \
                                               {df.n_user_subreddits.mean()}, \
                                               {df.n_user_subreddits.max()}')

    # Save dataset plots
    plot_aggregates(df, save_file=str(FIG_PATH/'pushshift_descriptives.png')) # Define filename above

    # Plot dataset size
    plot_size_curve(sizedict, save_file=str(FIG_PATH/'pushshift_size.png')) # Define filename above


if __name__ == '__main__':
    preprocess()


