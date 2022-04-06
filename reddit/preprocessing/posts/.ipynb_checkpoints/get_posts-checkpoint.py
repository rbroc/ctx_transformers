import sqlite3
from pathlib import Path
import gzip
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=10000,
                    help='Number of examples per output file')
parser.add_argument('--dataset-size', type=int, default=10000,
                    help='Number of examples per output file')
parser.add_argument('--perc-train', type=float, default=.9,
                    help='Percentage of posts in training set')
parser.add_argument('--perc-test', type=float, default=.1,
                    help='Percentage of posts in test set')


DATA_PATH = Path('..') / '..' / 'data'
DB_PATH = DATA_PATH / 'databases'
META_PATH = DATA_PATH / 'meta'
POSTS_PATH = DATA_PATH / 'json' / 'posts'


conn = sqlite3.connect(str(DB_PATH / 'posts.db'))
cur = conn.cursor()


def _save_example_dicts(posts, PATH, idxs):
    ''' Takes list of posts, makes dictionary and stores
    Args:
        posts (lst): list of dictionary examples
        path (Path or str): output path
        idxs (lst): indices
    '''
    for i in range(len(idxs)-1):
        start_idx = idxs[i]
        end_idx = idxs[i+1]
        d = [dict(zip(['id','text','comments', 'score'],
                       p))
                   for p in posts[start_idx:end_idx]]
        ofile = PATH / f'batch_{i+1}.json.gz'
        with gzip.open(str(ofile), 'w') as fh:
            fh.write(json.dumps(d).encode('utf-8'))


def make_aggregate_dataset(dataset_size=1000000, batch_size=10000,
                           perc_train=.9, perc_test=.1):
    ''' Create simple dataset with posts and metrics
    Args:
        n_posts (int): number of posts per user
        batch_size (int): how many posts per output file
        perc_train (float): percentage of posts in training set
        perc_train (float): percentage of posts in test set
    '''
    
    
    # Get posts
    cur.execute(f'''SELECT selftext, num_comments, score FROM posts
                    ORDER BY RANDOM()
                    LIMIT {str(dataset_size)}''')
    posts = list(cur.fetchall())
    posts = [(i,) + p for i,p in enumerate(posts)]
    
    # Make train and test splits
    tr_posts = posts[:int(len(posts)*perc_train)]
    ts_posts = posts[int(len(posts)*perc_train):
                     int(len(posts)*perc_train)+int(len(posts)*perc_test)]
    
    # Idxs
    tr_idx = list(np.arange(0, len(tr_posts), batch_size)) + [len(tr_posts)]
    ts_idx = list(np.arange(0, len(ts_posts), batch_size)) + [len(ts_posts)]
    
    # Make paths
    TRAIN = POSTS_PATH / f'{dataset_size}_posts' / 'train'
    TEST = POSTS_PATH / f'{dataset_size}_posts' / 'test'
    TRAIN.mkdir(exist_ok=True, parents=True)
    TEST.mkdir(exist_ok=True, parents=True)
    
    # Save dicts
    _save_example_dicts(tr_posts, TRAIN, tr_idx)
    _save_example_dicts(ts_posts, TEST, ts_idx)
    
    
if __name__=='__main__':
    args = parser.parse_args()
    make_aggregate_dataset(args.dataset_size, args.batch_size,
                           args.perc_train, args.perc_test)
    conn.close()