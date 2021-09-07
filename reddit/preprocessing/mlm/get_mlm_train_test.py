import sqlite3
import argparse
from pathlib  import Path
import random
import json
import gzip
from reddit.utils import stringify

parser = argparse.ArgumentParser()
parser.add_argument('--min-posts', type=int, default=11, 
                    help='Keeps only groups with >= this number of posts')
parser.add_argument('--test-prop', type=float, default=.1,
                    help='Size of test set relative to total '
                         'number of subreddits/authors')


DATA_PATH = Path('..') / '..' / 'data'
DB_PATH = DATA_PATH / 'databases'
META_PATH = DATA_PATH / 'meta'


conn = sqlite3.connect(str(DB_PATH / 'posts.db'))
cur = conn.cursor()


def _split(test_prop):
    all_ids = cur.fetchall()
    all_ids = [i[0] for i in all_ids]
    test_ids = random.sample(all_ids, 
                             int(len(all_ids)*test_prop))
    train_ids = list(set(all_ids) - set(test_ids))
    return all_ids, train_ids, test_ids


def _rowid_query(users_list, srs_list, split='train'):
    print(f'*** Getting {split} ids... ***')
    cur.execute(f'''SELECT ROWID, author, subreddit 
                    FROM posts
                    WHERE author IN {stringify(users_list)} AND
                          subreddit IN {stringify(srs_list)}''')
    posts = cur.fetchall()
    ids = [r[0] for r in posts]
    nr_auths = len(set([r[1] for r in posts]))
    nr_sr = len(set([r[2] for r in posts]))
    print(f'{nr_auths} authors in {split} set')
    print(f'{nr_sr} subreddits in {split} set')
    print(f'{len(posts)} posts in {split} set')
    return ids


def train_test_split(min_posts=11, 
                     test_prop=.1):
    
    print('*** Sampling users ***')
    cur.execute(f'''SELECT DISTINCT(author) 
                    FROM posts
                    GROUP BY author 
                    HAVING COUNT(DISTINCT(id))>= {min_posts}''')
    users, users_train, users_test = _split(test_prop)
    
    print('*** Sampling subreddits ***')
    cur.execute(f'''SELECT DISTINCT(subreddit) 
                    FROM posts 
                    GROUP BY subreddit 
                    HAVING COUNT(DISTINCT(id))>= {min_posts}''')
    srs, srs_train, srs_test = _split(test_prop)
    
    print('*** Retrieving train target ids... ***')
    train_ids = _rowid_query(users_train, srs_train)
    print(f'*** Retrieving test target ids... ***')
    test_ids = _rowid_query(users_test, srs_test, split='test')
    
    print('*** Saving ids as json... ***')
    json_ids = {'train_target_rowids': train_ids, 
                'test_target_rowids': test_ids,
                'train_authors': users_train,
                'test_authors': users_test,
                'train_subreddits': srs_train,
                'test_subreddits': srs_test}
    ofile = str(META_PATH/f'{min_posts}_mlm_splits.json.gz')
    with gzip.open(ofile,'wt') as fh: 
        fh.write(json.dumps(json_ids)) 
        
    
if __name__=='__main__':
    args = parser.parse_args()
    train_test_split(args.min_posts,
                     args.test_prop)
    conn.close()