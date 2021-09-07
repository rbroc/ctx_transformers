import sqlite3
import pandas as pd
from pathlib import Path
from datetime import date
import gzip
import json
import argparse
from reddit.utils import stringify
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-n-posts', type=int, 
                    default=3, help='Nr posts per author')
parser.add_argument('--batch-size', type=int, default=10000,
                    help='Number of examples per output file')
parser.add_argument('--perc-train', type=float, default=.9,
                    help='Percentage of authors in training set')
parser.add_argument('--perc-test', type=float, default=.1,
                    help='Percentage of authors in test set')


DATA_PATH = Path('..') / '..' / 'data'
DB_PATH = DATA_PATH / 'databases'
META_PATH = DATA_PATH / 'meta'
AGG_PATH = DATA_PATH / 'json' / 'agg'


conn = sqlite3.connect(str(DB_PATH / 'posts.db'))
cur = conn.cursor()


def _query_agg_by_author(column, round_to, key, subset, agg='AVG'):
    ''' Queries by-author aggregate for a column 
    Args:
        column (str): which column to fetch averages from
        round_to (int): how many decimals to round to
        key (str): how to name the key in the output dictionary
        subset (list): author subset
    '''
    query = f'''SELECT author, ROUND({agg}({column}), {round_to}) FROM posts
                WHERE author IN {stringify(subset)}
                GROUP BY author
             '''
    cur.execute(query)
    res = cur.fetchall()
    resdct = {a: {key: v} for a,v in res}
    return resdct


def _get_date(which):
    ''' Get earliest or latest date in dataset 
    Args:
        which (str): sql aggregate function
    '''
    query = f'''SELECT {which}(date(created_utc,\'unixepoch\')) 
                FROM posts'''
    cur.execute(query)
    date = [int(s) for s in cur.fetchall()[0][0].split('-')]
    return date


def _zip_dicts(ds):
    ''' Merges dictionaris with common keys'''
    for idx, k in enumerate(ds[0].keys()):
        
        ad = {'author': k,
              'author_id': idx}
        for d in ds:
            ad.update(d[k])
        yield ad
    
    
def _sample_authors(author_list, n):
    ''' Sample n posts from a certain author
    Args:
        author_list (list): list with author ids
        n (int): how many context posts to sample for each author
    '''
    ds = {}
    for idx, author in enumerate(author_list):
        if (idx + 1) % 10000 == 0:
            print(f'Sampling {idx + 1}')
        cur.execute(f''' SELECT subreddit, selftext 
                         FROM posts
                         WHERE author = '{author}'
                         ORDER BY RANDOM() LIMIT {n}
                     ''')     
        res = cur.fetchall()
        subs = [r[0] for r in res]
        stext = [r[1] for r in res]
        ds[author] = {'author_id': idx,
                      'subreddits': subs,
                      'posts': stext}
    return ds


def _save_example_dicts(d, outpath, batch_size=10000):
    ''' Takes list of example dictionaries and stores in 
        json files with batch_size examples per file
    Args:
        d (lst): list of dictionary examples
        outpath (pathlib.Path): output path for json files
        size (int): number of examples in the dataset
        batch_size (int): number of examples per file
    '''
    idxs = list(np.arange(0, len(d), batch_size)) + [len(d)]
    for nidx in range(len(idxs[:-1])):
        ofile = outpath / f'batch_{nidx+1}.json.gz'
        with gzip.open(ofile, 'w') as fh:
            fh.write(json.dumps(d[idxs[nidx]:idxs[nidx+1]]).encode('utf-8'))


def make_aggregate_dataset(n_posts=10, batch_size=10000,
                          perc_train=.9, perc_test=.1):
    ''' Create datasets with n_posts posts per author and 
        aggregate metrics as labels (avg number of upvotes,
        comments, etc.). Which metrics is hard-coded right now.
    Args:
        n_posts (int): number of posts per user
        batch_size (int): how many posts per output file
        perc_train (float): percentage of authors in training set
        perc_train (float): percentage of authors in test set
    '''
    print('*** Splitting train/test authors ***')
    cur.execute('''SELECT DISTINCT(author) FROM posts
                   ORDER BY RANDOM()''')
    auths = [i[0] for i in cur.fetchall()]
    tr_auths = auths[:int(len(auths)*perc_train)]
    ts_auths = auths[int(len(auths)*perc_train):
                     int(len(auths)*perc_train)+int(len(auths)*perc_test)]
    
    print('*** Fetching average submission metrics ***')
    tr_upv = _query_agg_by_author('score', 5, 'avg_score', tr_auths, 'AVG')
    tr_com = _query_agg_by_author('num_comments', 5, 'avg_comm', tr_auths, 'AVG')
    ts_upv = _query_agg_by_author('score', 5, 'avg_score', ts_auths, 'AVG')
    ts_com = _query_agg_by_author('num_comments', 5, 'avg_comm', ts_auths, 'AVG')

    print('*** Fetching average number of posts per day ***')
    max_date = _get_date('MAX')
    min_date = _get_date('MIN')
    n_days = (date(*max_date) - date(*min_date)).days
    tr_nposts = _query_agg_by_author('id', 0, 'n_posts', tr_auths, 'COUNT')
    ts_nposts = _query_agg_by_author('id', 0, 'n_posts', ts_auths, 'COUNT')
    tr_nposts_per_day = {k: {'n_posts': v['n_posts'] / n_days} 
                        for k,v in tr_nposts.items()}
    ts_nposts_per_day = {k: {'n_posts': v['n_posts'] / n_days} 
                        for k,v in ts_nposts.items()}
    
    print('*** Fetching reference posts per author ***')
    tr_posts = _sample_authors(tr_auths, n_posts)
    ts_posts = _sample_authors(ts_auths, n_posts)
    
    print('*** Merging posts with metrics ***')
    gen_train = _zip_dicts([tr_posts, tr_upv, tr_com, tr_nposts_per_day])
    gen_test = _zip_dicts([ts_posts, ts_upv, ts_com, ts_nposts_per_day])
    
    print('*** Saving json files ***')
    for split, gen in zip(['train', 'test'], 
                          [gen_train, gen_test]):
        lst = [e for e in gen]
        OUTPATH = AGG_PATH / f'{n_posts}_random' / split
        OUTPATH.mkdir(exist_ok=True, parents=True)
        _save_example_dicts(lst, OUTPATH, batch_size)
    
    
if __name__=='__main__':
    args = parser.parse_args()
    make_aggregate_dataset(args.n_posts, args.batch_size,
                           args.perc_train, args.perc_test)
    conn.close()