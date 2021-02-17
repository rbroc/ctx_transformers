import zstandard as zstd
import lzma
import requests
import os
from pathlib import Path
import json
import itertools
from IPython.display import clear_output
import pandas as pd


# Directory params
HOME_DIR = os.path.expanduser('~')
SAVE_DIR = Path(HOME_DIR) / 'tmp'
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# Requests url
URL = 'https://files.pushshift.io/reddit/submissions/'


# Define useful functions
def filter_line(ldict, posts, target_fields):
    if (ldict['subreddit'] in srlist) and \
        (ldict['selftext'] not in ['', '[deleted]', '[removed]']) and \
        (ldict['author'] != '[deleted]') and \
        (ldict['is_self'] == True):
        ldict = {k: ldict[k] for k in target_fields}
        posts.append(ldict)
        return ldict, posts

def save_file(posts, year, month, idx, fprefix):
    idx += 1
    clear_output(wait=True)
    df = pd.DataFrame(posts)
    df.to_csv(f'../raw/pushshift/{year}_{month}_{idx*100000}.txt', 
            sep='\t', index=False)
    print(f'Saving {fprefix} {idx*100000}...')
    posts = []
    return posts, idx


# Pushshift dump params
years = [str(i) for i in [2018]]
months = ['0' + str(i) for i in range(1, 10)] + [str(i) for i in [10, 11, 12]]
ym_comb = itertools.product(years, months)


# Filtering params
srlist = list(pd.read_csv('../misc/top_1000_subreddits.txt', sep='\t').real_name)
target_fields = ['author', 'created_utc', 'domain', 'id', 
                'num_comments', 'num_crossposts', 'score', 
                'selftext', 'subreddit', 'subreddit_id',
                'title']

for year, month in ym_comb:
    # Request
    fprefix = ''.join(['RS', '_', year, '-', month])
    furl = ''.join([URL, fprefix])
    cformat = '.zst'
    r = requests.get(furl + cformat, stream=True)
    if r.status_code == 404:
        cformat = '.xz'
        r = requests.get(furl + '.xz', stream=True)

    # Download and save the file
    if not os.path.exists(SAVE_DIR / (fprefix + cformat)):
        with open(SAVE_DIR / (fprefix + cformat), 'wb') as f:
            for idx, chunk in enumerate(r.iter_content(chunk_size=16384)):
                if (idx != 0) and (idx % 1000 == 0):
                    clear_output(wait=True)
                    print(f'Writing file {(fprefix + cformat)}: chunk {idx}')
                f.write(chunk)
            print('Done writing!')
    else:
        print(f'{fprefix} already downloaded!')

    # Decompress, filter, and save filtered version
    posts = []
    idx = 0

    # routine for xz format 
    if cformat == '.xz':
        with lzma.open(SAVE_DIR / (fprefix + cformat), mode='rt') as fh:
            for line in fh:
                ldict = json.loads(line)
                posts = filter_line(ldict, posts, target_fields)
                if len(posts) == 100000:
                    posts, idx = save_file(posts, year, month, idx, fprefix)
            if posts != []:
                save_idx = idx * 100000 + len(posts)
                pd.DataFrame(posts).to_csv(f'../raw/pushshift/{year}_{month}_{save_idx}.txt', 
                            sep='\t', index=False)

    # Special routine for zst format
    elif cformat == '.zst':
        with open(SAVE_DIR / (fprefix + cformat), 'rb') as fh:
            dcmp = zstd.ZstdDecompressor()
            buffer = ''
            with dcmp.stream_reader(fh) as reader:
                while True:
                    chunk = reader.read(8192).decode()
                    if not chunk:
                        if posts != []:
                            save_idx = idx * 100000 + len(posts)
                            pd.DataFrame(posts).to_csv(f'../raw/pushshift/{year}_{month}_{save_idx}.txt', 
                                    sep='\t', index=False)
                        break
                    lines = (buffer + chunk).split('\n')
                    for line in lines[:-1]:
                        ldict = json.loads(line)
                        posts = filter_line(ldict, posts, target_fields)
                        if len(posts) == 100000:
                            posts, idx = save_file(posts, year, month, idx, fprefix)
                    buffer = lines[-1]          
 
    # Remove
    os.remove(SAVE_DIR / (fprefix + cformat))