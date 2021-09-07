import sqlite3
from pathlib import Path
import gzip
import json
from reddit.utils import stringify
import argparse
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--min-posts', type=int, default=11, 
                    help='Keeps only groups with >= this number of posts')
parser.add_argument('--n-context', type=int, 
                    default=10, help='Nr context posts')
parser.add_argument('--train-size', type=int, default=200000,
                    help='Number of targets to sample for training set')
parser.add_argument('--test-size', type=int, default=100000,
                    help='Number of targets to sample for test set')
parser.add_argument('--batch-size', type=int, default=10000,
                    help='Number of examples per output file')


DATA_PATH = Path('..') / '..' / 'data'
DB_PATH = DATA_PATH / 'databases'
META_PATH = DATA_PATH / 'meta'
MLM_PATH = DATA_PATH / 'json' / 'mlm'

conn = sqlite3.connect(str(DB_PATH / 'posts.db'))
cur = conn.cursor()


def _sample_context(id_list=None, 
                    grouping_list=None,
                    by='author',
                    retrieve_list=None,
                    n=3):
    ''' Sample context posts given a list of target group ids 
        or a list of ids
    Args:
        id_list (list): list of post ids (if by not random)
        grouping_list (list): list with author or subreddit for each 
            post id in id_list (if by!='random')
        by (str): whether grouping_list is authors, subreddits or random
        retrieve_list (lst): if by='random', this is the list of ids to 
            retrieve
        n (int): how many context posts to sample for each target
    '''
    ds = []
    if by != 'random':
        for idx, (target, group) in enumerate(zip(id_list, 
                                                  grouping_list)):
            if (idx + 1) % 10000 == 0:
                print(f'Sampling {idx + 1}')
            cur.execute(f''' SELECT ROWID, id, author, subreddit, selftext 
                             FROM posts
                             WHERE {by} = '{group}' AND 
                             ROWID <> '{target}'
                             ORDER BY RANDOM() LIMIT {n}
                         ''')     
            ds += cur.fetchall()
    else:
        for idx, r_l in enumerate(retrieve_list):
            if (idx + 1) % 10000 == 0:
                print(f'Sampling {idx + 1}')
            cur.execute(f''' SELECT ROWID, id, author, subreddit, selftext
                             FROM posts
                             WHERE ROWID IN {stringify(r_l)}
                        ''')       
            ds += cur.fetchall()
    return ds
        

def _batch_and_add_id(lst, n_context):
    ''' Group query results in batches of size n context.
        Also adds an index as unique identifier to the example.
    Args:
        lst (list): query result list
        n_context (int): how many items to group together
    '''
    batched = [lst[i:i+n_context] 
               for i in np.arange(0, len(lst), n_context)]
    batched = [(idx+1, a) 
               for idx, a in enumerate(batched)]
    return batched
    

def _make_example_dicts(target_ids, contexts, target_d):
    ''' Make example dictionary (to dump onto json file) '''
    mlm_dict = []
    for trg, ctx in zip(target_ids, 
                        contexts):
        mlm_dict.append({'example_id': ctx[0],
                         'target': [target_d[trg]['selftext'].strip('\n')],
                         'target_author': [target_d[trg]['author']],
                         'target_subreddit': [target_d[trg]['subreddit']],
                         'target_id': [target_d[trg]['post_id']],
                         'target_rowid': [trg],
                         'context': [c[-1].strip('\n') for c in ctx[1]],
                         'context_authors': [c[2] for c in ctx[1]],
                         'context_subreddits': [c[3] for c in ctx[1]], 
                         'context_ids': [c[1] for c in ctx[1]],
                         'context_rowids': [c[0] for c in ctx[1]],
                         'author_overlap': int(target_d[trg]['author'] in \
                                               [c[2] for c in ctx[1]]),
                         'subreddit_overlap': int(target_d[trg]['subreddit'] in \
                                                  [c[3] for c in ctx[1]])})
    return mlm_dict


def _save_example_dicts(d, outpath, size, batch_size=10000):
    ''' Takes list of example dictionaries and stores in 
        json files with batch_size examples per file
    Args:
        d (lst): list of dictionary examples
        outpath (pathlib.Path): output path for json files
        size (int): number of examples in the dataset
        batch_size (int): number of examples per file
    '''
    idxs = list(np.arange(0, size, batch_size)) + [size]
    for nidx in range(len(idxs)-1):
        ofile = outpath / f'batch_{nidx+1}.json.gz'
        with gzip.open(ofile, 'w') as fh:
            fh.write(json.dumps(d[idxs[nidx]:idxs[nidx+1]]).encode('utf-8')) 

def _json_save_loop(g_type, g_dict, n_context, split, size, batch_size):
    OUTPATH = MLM_PATH / f'{n_context}context_large' / g_type / split # replaced random with large
    OUTPATH.mkdir(exist_ok=True, parents=True)
    _save_example_dicts(g_dict, OUTPATH, size, batch_size)
    

def make_examples(n_context=3, split='train', 
                  size=200000, batch_size=10000, 
                  min_posts=5):
    ''' Make datasets with target/context pairs for 
        all grouping type (by author, by subreddit, random,
        no contex)
    Args:
        n_context (int): number of context posts
        split (str): 'train' or 'test'
        size (int): size of the dataset
        batch_size (int): number of examples per output file
    '''
    try:
        assert batch_size <= size
    except:
        raise ValueError('batch_size must be smaller than dataset size')
    
    ofile = str(META_PATH/f'{min_posts}_mlm_splits.json.gz')
    target_meta = json.load(gzip.open(ofile))
    target_string = stringify(target_meta[f'{split}_target_rowids'])
    
    print(f'*** Querying targets from db ***')
    cur.execute(f'''SELECT ROWID, id, author, subreddit, selftext 
                    FROM posts
                    WHERE ROWID IN {target_string}''')
    targets = cur.fetchall()
    target_d = {r[0]: {'post_id': r[1],
                       'subreddit': r[3],
                       'author': r[2],
                       'selftext': r[4]} for r in targets}

    print(f'*** Sampling targets ***')
    ids = random.choices(list(target_d.keys()), k=size)
    authors = [target_d[i]['author'] for i in ids]
    srs = [target_d[i]['subreddit'] for i in ids]
    
    print('*** Sampling author contexts *** ')
    author_ds = _sample_context(id_list=ids, 
                                grouping_list=authors, 
                                by='author', n=n_context)
    author_ds_batched = _batch_and_add_id(author_ds, n_context)
    author_dict = _make_example_dicts(ids, author_ds_batched, target_d)
    _json_save_loop('author', author_dict, n_context, split, size, batch_size)
    del author_ds_batched, author_ds, authors
    
    print('*** Saving no-context *** ')
    single_dict = []
    for di in author_dict:
        single_dict.append({k: di[k] 
                            for k in 
                            ['example_id', 
                             'target', 'target_author',
                             'target_subreddit', 'target_id', 
                             'target_rowid']})
    _json_save_loop('single', single_dict, n_context, split, size, batch_size)
    del author_dict, single_dict
    
    print('*** Sampling random contexts ***')
    if split == 'train':
        exclude_authors = target_meta['test_authors']
        exclude_srs = target_meta['test_subreddits']
    if split == 'test':
        exclude_authors = target_meta['train_authors']
        exclude_srs = target_meta['train_subreddits']
    cur.execute(f'''SELECT ROWID FROM posts 
                    WHERE author NOT IN {stringify(exclude_authors)} AND
                          subreddit NOT IN {stringify(exclude_srs)}
                ''')
    rands = [r[0] for r in cur.fetchall()]
    r_contexts = []
    for t_id in ids:
        while True:
            r_idxs = [int(len(rands) * random.random()) 
                      for _ in range(n_context)]
            r_sample = [rands[i] for i in r_idxs]
            if (len(set(r_sample))==n_context) and (t_id not in r_sample):
                r_contexts.append(r_sample)
                break   
    random_ds = _sample_context(by='random', n=n_context, 
                                retrieve_list=r_contexts)
    random_ds_batched = _batch_and_add_id(random_ds, n_context)
    random_dict = _make_example_dicts(ids, random_ds_batched, target_d)
    _json_save_loop('random', random_dict, n_context, split, size, batch_size)
    del (random_ds, rands, random_dict, random_ds_batched, 
         exclude_authors, exclude_srs)
    

    print('*** Sampling subreddit contexts *** ')
    sr_ds = _sample_context(id_list=ids, 
                            grouping_list=srs, 
                            by='subreddit', n=n_context)
    sr_ds_batched = _batch_and_add_id(sr_ds, n_context)
    sr_dict = _make_example_dicts(ids, sr_ds_batched, target_d)
    _json_save_loop('subreddit', sr_dict, n_context, split, size, batch_size)
    del sr_ds_batched, sr_dict, sr_ds, srs
    

if __name__=='__main__':
    args = parser.parse_args()
    make_examples(args.n_context, 'train', args.train_size, 
                  args.batch_size, args.min_posts)
    make_examples(args.n_context, 'test', args.test_size, 
                  args.batch_size, args.min_posts)
    conn.close()
    