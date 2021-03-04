import zstandard as zstd
import lzma
import requests
import os
from pathlib import Path
import json
import itertools
import pandas as pd


# Directory params for download
DOWNLOAD_DIR = Path('..') / 'data' / 'tmp'
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR = Path('..') / 'data' / 'raw'
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# Url for requests
URL = 'https://files.pushshift.io/reddit/submissions/'


# Pushshift files params
years = [str(i) for i in [2018, 2019]]
months = ['0' + str(i) for i in range(1, 10)] + [str(i) for i in [10, 11, 12]]
ym_comb = itertools.product(years, months)

# Filtering params
target_fields = ['author', 'created_utc', 'domain', 'id', 
                'num_comments', 'num_crossposts', 'score', 
                'selftext', 'subreddit', 'subreddit_id',
                'title']

# Define submission filtering function
def filter_line(ldict, posts, target_fields):
    if (ldict['over_18'] is False) and \
        (ldict['selftext'] not in ['', '[deleted]', '[removed]']) and \
        (ldict['author'] != '[deleted]') and \
        (ldict['is_self'] == True):
        ldict = {k: ldict[k] for k in target_fields}
        posts.append(ldict)
        return ldict, posts

# Define logging function
def save_file(posts, year, month, idx, fprefix):
    idx += 1
    df = pd.DataFrame(posts)
    outfile = SAVE_DIR / f'{year}_{month}_{idx*100000}.txt'
    df.to_csv(outfile, sep='\t', index=False, compression='gzip')
    print(f'Saving {fprefix} {idx*100000}...')
    posts = []
    return posts, idx

# Main function
def download_and_extract():
    ''' Downloads Reddit dump, filters posts and saves as tsv '''

    for year, month in ym_comb:
        # Request
        fprefix = ''.join(['RS', '_', year, '-', month])
        furl = ''.join([URL, fprefix])
        cformat = '.zst'
        r = requests.get(furl + cformat, stream=True)
        if r.status_code == 404:
            cformat = '.xz'
            r = requests.get(furl + '.xz', stream=True)

        # Download and save
        if not os.path.exists(DOWNLOAD_DIR / (fprefix + cformat)):
            with open(DOWNLOAD_DIR / (fprefix + cformat), 'wb') as f:
                for idx, chunk in enumerate(r.iter_content(chunk_size=16384)):
                    if (idx != 0) and (idx % 1000 == 0):
                        print(f'Writing file {(fprefix + cformat)}: chunk {idx}')
                    f.write(chunk)
                print('Done writing!')
        else:
            print(f'{fprefix} already downloaded!')

        # Decompress filter and save file
        posts = []
        idx = 0

        # xz format 
        if cformat == '.xz':
            with lzma.open(DOWNLOAD_DIR / (fprefix + cformat), mode='rt') as fh:
                for line in fh:
                    ldict = json.loads(line)
                    posts = filter_line(ldict, posts, target_fields)
                    if len(posts) == 100000:
                        posts, idx = save_file(posts, year, month, idx, fprefix)
                if posts != []:
                    save_idx = idx * 100000 + len(posts)
                    outfile = SAVE_DIR / f'{year}_{month}_{save_idx}.txt'
                    pd.DataFrame(posts).to_csv(outfile, sep='\t', index=False, 
                                               compression='gzip')

        # zst format
        elif cformat == '.zst':
            with open(DOWNLOAD_DIR / (fprefix + cformat), 'rb') as fh:
                dcmp = zstd.ZstdDecompressor()
                buffer = ''
                with dcmp.stream_reader(fh) as reader:
                    while True:
                        chunk = reader.read(8192).decode()
                        if not chunk:
                            if posts != []:
                                save_idx = idx * 100000 + len(posts)
                                outfile = SAVE_DIR / f'{year}_{month}_{save_idx}.txt'
                                pd.DataFrame(posts).to_csv(outfile, sep='\t', 
                                                           index=False, 
                                                           compression='gzip')
                            break
                        lines = (buffer + chunk).split('\n')
                        for line in lines[:-1]:
                            ldict = json.loads(line)
                            posts = filter_line(ldict, posts, target_fields)
                            if len(posts) == 100000:
                                posts, idx = save_file(posts, year, month, idx, fprefix)
                        buffer = lines[-1]          
    
        os.remove(DOWNLOAD_DIR / (fprefix + cformat))

    os.rmdir(DOWNLOAD_DIR)


if __name__ == '__main__':
    download_and_extract()