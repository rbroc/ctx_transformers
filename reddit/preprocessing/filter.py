import pandas as pd
import numpy as np
import os
from pathlib import Path
import argparse
import glob
import csv
import gzip
import random
import seaborn as sns
from matplotlib import pyplot as plt
from selectolax.parser import HTMLParser
from markdown import markdown
from pandarallel import pandarallel

# Define paths
DATA_PATH = Path('..') / 'data'
RAW_PATH = DATA_PATH / 'raw'

PROCESSED_PATH = DATA_PATH / 'filtered'
PROCESSED_PATH.mkdir(exist_ok=True)
AUTHOR_PATH = DATA_PATH / 'users'
AUTHOR_PATH.mkdir(exist_ok=True)
FIG_PATH = DATA_PATH / 'figures'
FIG_PATH.mkdir(exist_ok=True)

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--min-posts', type=int, default=5,
                    help='Minimum number of posts per user')
parser.add_argument('--min-subreddits', type=int, default=5,
                    help='Minimum number of subreddits per user')
parser.add_argument('--num-cores', type=int, default=3,
                    help='Number of parallel cores used in apply call')


# Util function to remove markdown syntax
def markdown_to_text(md):
    html = markdown(md)
    tree = HTMLParser(html)
    return tree.body.text()


def filter(min_posts=5, min_subreddits=5, num_cores=3):
    ''' Runs preprocessing on the dataset (filtering duplicates,
        filtering by minimum number of posts per user, removing 
        non-English posts) 
    Args:
        min_posts (int): minimum number of posts per user
        min_subreddits (int): mininum number of subreddits to 
            which the user has to have contributed    
    '''
    pandarallel.initialize(nb_workers=num_cores)

    fs = glob.glob(str(RAW_PATH/'*'))
    n_posts = []
    n_subreddits = []
    subreddits = set()

    # Split global files into files by author
    print(f'Splitting files into single-author files...')
    dropped = 0
    for fidx, f in enumerate(fs):
        print(f'{fidx+1} of {len(fs)}')
        with gzip.open(f, 'rt') as ifile:
            rdr = csv.reader(ifile, delimiter='\t')
            for row in rdr:
                if len(row) == 8:
                    ofile = AUTHOR_PATH / f'{row[0]}.txt' 
                    with open(ofile, 'a') as ofh:
                        ofh.write('\t'.join(row))
                else:
                    dropped += 1
        print(f'\t{dropped} invalid rows so far')
        os.remove(f)
    os.rmdir(RAW_PATH)
    
    # Filter author files
    afs = glob.glob(str(AUTHOR_PATH/'*'))
    random.shuffle(afs)
    print(f'Filtering author files...')
    group_count = 0
    n_groups = 1
    cols = ['author', 'created_utc', 'id', 
            'num_comments', 'score', 'selftext',
            'subreddit', 'title', 'lang']

    for fidx, f in enumerate(afs):
        # Read file
        adf = pd.read_csv(f, sep='\t', header=False)
        adf.columns = cols
        # Remove duplicates and count posts/subreddits
        adf = adf.drop_duplicates(subset=['selftext'])
        nps = adf.shape[0]
        nss =  adf.subreddit.nunique()

        if (nps >= min_posts) and (nss >= min_subreddits):
            adf['selftext'] = adf['selftext'].parallel_apply(markdown_to_text)
            adf['selftext'].replace('[\s]+', ' ', regex=True, inplace=True)
            # Log user metrics
            adf['n_user_posts'] = nps
            adf['n_user_subreddits'] = nss
            n_posts.append(nps)
            n_subreddits.append(nss)
            subreddits = subreddits.union(set(adf.subreddit.tolist()))
            # Append to group_df
            if group_count == 0:
                group_df = adf
            else:
                group_df = pd.concat([group_df, 
                                      adf], ignore_index=True)
            group_count += 1
            # Save if 1000 users
            if group_count % 1000 == 0:
                outfile = PROCESSED_PATH / f'{1000*n_groups}.txt.gz'
                group_df['user_id'] = group_df['author']\
                    .map(range(1000*(n_groups-1), 1000*n_groups))
                group_df.to_csv(outfile, sep='\t', compression='gzip',
                                index=False)
                n_groups += 1
                group_count = 0
        os.remove(f)

    # Print dataset features
    print(f'''\nThere are {len(n_posts)} users, 
                          {np.sum(n_posts)} posts,
                          {len(set(subreddits))} subreddits''')

    # Get aggregates
    pmetrics = [getattr(np, m)(n_posts) for m in ['max', 'min', 'mean']]
    smetrics = [getattr(np, m)(n_subreddits) for m in ['max', 'min', 'mean']]
    print(f'Min, avg, max posts per user: {pmetrics}')
    print(f'Min, avg, max subreddits per user: {smetrics}')

    # Plot aggregates and save
    fname_agg = str(FIG_PATH/'aggregates.png')
    f, ax = plt.subplots(nrows=2)
    vars = [n_posts, n_subreddits]
    xlabs = ['# posts', '# subreddits']
    for i in range(2):
        sns.histplot(x=vars[i], ax=ax[i], bins=100)
        ax[i].set_xlabel(xlabs[i])
        ax[i].set_ylabel('# users')
        ax[i].legend('')
    plt.tight_layout()
    plt.savefig(fname_agg)


if __name__ == '__main__':
    args = parser.parse_args()
    filter(args.min_posts, args.min_subreddits, args.num_cores)