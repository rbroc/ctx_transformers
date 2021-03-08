import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
import argparse
import json
from reddit.utils import read_files

DATA_PATH = Path('..') / 'data'
PROCESSED_PATH = DATA_PATH  /'filtered'
TRIPLET_PATH = DATA_PATH / 'triplet'
TRIPLET_PATH.mkdir(parents=True, exist_ok=True)
DATASET_PATH = DATA_PATH / 'datasets' / 'triplet'
DATASET_PATH.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0,
                    help='Seed for random call')

def _pick_examples(user_df):
    ''' Picks one example per user'''
    idx = np.random.choice(user_df.index, 1)[0]
    return idx, user_df.loc[idx,:]

def _extract_and_drop_examples(df):
    idx, ex_df = df.groupby('user_id', 
                            as_index=False).apply(_pick_examples)
    df = df.drop(idx, axis=0)
    return df, ex_df

def _make_json(sub_df):
    adict = {}
    for u in sub_df.user_id.unique():
        u_df = sub_df[sub_df['target_user']==u]
        anchor = u_df[u_df['example_type']=='anchor']
        pos = u_df[u_df['example_type']=='positive']
        neg = u_df[u_df['example_type']=='negative']
        anchor_sr = anchor['subreddit'].unique().tolist()
        unique_sr = u_df.subreddit.nunique()
        adict[u] = {'anchor': anchor['selftext'].tolist(),
                    'pos': pos['selftext'].iloc[0],
                    'neg': neg['selftext'].iloc[0],
                    'n_anchor': anchor.shape[0],
                    'neg_author': neg['author'].iloc[0],
                    'anchor_subreddits': anchor_sr,
                    'pos_subreddit': pos['subreddit'].iloc[0],
                    'neg_subreddit': neg['subreddit'].iloc[0],
                    'author_unique_subreddits': unique_sr}
    return adict


def make_triplet_examples(seed=0):
    ''' For each user, selects which posts are used as anchor,
        positive example, and negative example. Stores this info 
        in dataframes (splits into several chunks for ease of 
        processing and storage reasons) and in a json file
    Args:
        seed (int): seed for np.random.seed call
    '''
    df = read_files(PROCESSED_PATH, drop_duplicates=False)
    
    # Make examples
    np.random.seed()
    print('Extracting per-user negative examples...')
    neg_df, df = _extract_and_drop_examples(df)
    print('Extracting per-user positive examples...')
    pos_df, df = _extract_and_drop_examples(df)
    df['example_type'] = 'anchor'
    neg_df['example_type'] = 'negative'
    pos_df['example_type'] = 'positive'
    users = df.user_id.unique()
    attempt = 0
    while True:
        attempt += 1
        print(f'Combining users and examples, attempt {attempt}...')
        np.random.shuffle(users)
        if all(users != neg_df['user_id']):
            break
    neg_df['target_user_id'] = users
    pos_df['target_user_id'] = pos_df['user_id']
    df['target_user_id'] = df['user_id']

    # Concatenate and save
    df = pd.concat([df, pos_df, neg_df], ignore_index=False)
    idxs = np.arange(0, users.shape[0], 10000)
    idxs = idxs.append(users.shape[0])
    for idx in idxs[:-1]:
        outfile = TRIPLET_PATH / f'triplet_{idx+1}.txt'
        json_outfile = TRIPLET_PATH / f'examples_{idx+1}.json'
        sub_df = df[df['author'].isin(df.author.unique()[idx:idx+1])]
        sub_df.to_csv(outfile, sep='\t', compression='gzip')
        d = _make_json(sub_df)
        with open(json_outfile) as fh:
            fh.write(json.dumps(d))
    del d, sub_df


if __name__=="__main__":
    args = parser.parse_args()
    make_triplet_examples(args.seed)