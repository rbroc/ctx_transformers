import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
import argparse
import json
import glob


DATA_PATH = Path('..') / 'data'
PROCESSED_PATH = DATA_PATH  /'filtered'
TRIPLET_PATH = DATA_PATH / 'triplet'
TRIPLET_PATH.mkdir(parents=True, exist_ok=True)
DATASET_PATH = DATA_PATH / 'datasets' / 'triplet'
DATASET_PATH.mkdir(parents=True, exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0,
                    help='Seed for random call')
parser.add_argument('--n-examples', type=int, default=1,
                    help='Number of negative/positive examples')


def _pick_user_examples(udf, n):
    ''' Picks examples to be used as neg/pos examples for one user
    Args:
        udf (pd.DataFrame): user-specific dataframe
        n (int): number of examples
    '''
    idx = np.random.choice(udf.index, n)[0]
    return idx, udf.loc[idx,:]


def pick_and_drop_examples(df, n):
    ''' Picks examples, pops them from main df, returns them separate df 
    Args:
        df (pd.DataFrame): global dataframe
        n (int): number of examples per user 
    '''
    idx, ex_df = df.groupby('user_id', as_index=False)\
        .apply(lambda x: _pick_user_examples(x, n))
    df = df.drop(idx, axis=0)
    return df, ex_df


def _make_user_dict(df, pos_df, neg_df):
    ''' Returns dictionary with info on user '''
    adict = {}
    for u in df.user_id.unique():
        anchor = df[df['target_author']==u]
        neg = neg_df[neg_df['target_author']==u]
        pos = pos_df[pos_df['target_author']==u]
        adict[u] = {'anchor': anchor['selftext'].tolist(),
                    'positive': pos['selftext'].tolist(),
                    'negative': neg['selftext'].tolist(),
                    'n_anchor': anchor.shape[0],
                    'n_positive': pos.shape[0],
                    'n_negative': neg.shape[0],
                    'anchor_subreddits': anchor['subreddit'].tolist(),
                    'positive_subreddit': pos['subreddit'].tolist(),
                    'negative_subreddit': neg['subreddit'].tolist(),
                    'negative_authors': neg['author'].tolist()}
    return adict


# Main function
def make_triplets(seed=0, n_examples=1):
    ''' For each user, selects which posts are used as anchor,
        positive example, and negative example. Stores this info 
        in dataframes (splits into several chunks for ease of 
        processing and storage reasons) and in a json file
    Args:
        seed (int): seed for np.random.seed call
    '''
    # Read file
    fs = glob.glob(str(PROCESSED_PATH/'*'))
    
    for f in fs:
        print(f'Reading {f}...')
        df = pd.read_csv(f, sep='\t', compression='gzip')

        # Get examples
        np.random.seed()
        neg_df, df = pick_and_drop_examples(df, n=n_examples)
        pos_df, df = pick_and_drop_examples(df, n=n_examples)
        df['example_type'] = 'anchor'
        neg_df['example_type'] = 'negative'
        pos_df['example_type'] = 'positive'
        attempt = 0

        # Match users and examples
        while True: 
            attempt += 1
            print(f'\tMatch users and examples, attempt {attempt}...')
            alist = df.author.unique.tolist()
            np.random.shuffle(alist)
            alist = [a for a in alist for _ in range(n_examples)]
            if all(alist != neg_df['user_id']):
                break
        neg_df['target_author'] = alist
        pos_df['target_author'] = pos_df['author']
        df['target_author'] = df['author']

        # Concatenate and save as json
        ofile_id = f.split('/')[-1].split('.')[0]
        ofile_json = TRIPLET_PATH / f'{ofile_id}.json'
        d = _make_user_dict(df, pos_df, neg_df)
        with open(ofile_json) as fh:
            fh.write(json.dumps(d))

if __name__=="__main__":
    args = parser.parse_args()
    make_triplets(args.seed)