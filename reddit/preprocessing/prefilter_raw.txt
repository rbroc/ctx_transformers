import pandas as pd
import glob
from pathlib import Path
import os

# Directory params for download
RAW_DIR = Path('..') / 'data' / 'raw'
RAW_DIR.mkdir(parents=True, exist_ok=True)

def filter_authors():
    print(f'Pre-filtering authors...')
    fs = glob.glob(str(RAW_DIR/'*'))
    unique_users = []

    # Fine valid users (>5 comments)
    for f in fs:
        try:
            df = pd.read_csv(f, sep='\t', compression='gzip')
            unique_users += df.author.unique().tolist()
        except:
            print(f'{f} failed!')
    unique_users = pd.Series(unique_users).value_counts()
    valid_users = unique_users[unique_users>10].index
    pd.DataFrame(valid_users,
                 columns=['authors']).to_csv('valid_users.txt', index=False,
                                             sep='\t', compression='gzip')
    
    # Read files again and filter
    tot_posts = 0
    for f in fs:
        try:
            df = pd.read_csv(f, sep='\t', compression='gzip')
            df = df[df['author'].isin(valid_users.tolist())]
            outfile = f.rstrip('.txt') + '_filtered.txt'
            df.to_csv(outfile, sep='\t', index=False, compression='gzip')
            tot_posts += df.shape[0]
            print(f'Total posts at {f}: {tot_posts}')
            os.remove(f)
        except:
            print(f'Filtering {f} failed!')
    

if __name__=="__main__":
    filter_authors()