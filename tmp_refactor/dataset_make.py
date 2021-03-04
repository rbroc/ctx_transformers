import json
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path
from .datasets import (split_dataset, stack_examples, 
                       plot_split_stat, save_dataset, 
                       plot_subreddit_distribution)

# Define useful variables
PROCESSED_PATH = Path('processed') / 'pushshift'
LABEL_PATH = Path('labels')
FIG_PATH = Path('figures')
DATASET_PATH = Path('datasets')
INPUT_FILENAME = PROCESSED_PATH / '500sr_encoded.txt'

# Initialize split utils
gsplit_main = GroupShuffleSplit(n_splits=1, train_size=.70, 
                                test_size=.15, random_state=0)
gsplit_small = GroupShuffleSplit(n_splits=1, train_size=.25, 
                                 test_size=.05, random_state=0)

# Define useful function for dataset creation
def _process_and_save(df, gsplit, figname, dsname, dev):
    dslist = split_dataset(df, gsplit, dev)
    plot_split_stat(*dslist, save=True, fname=str(FIG_PATH/figname))
    dslist = stack_examples(dslist)
    for d in dslist:
        save_dataset(d, path=str(DATASET_PATH / dsname), shard=True, 
                    compression='GZIP', n_shards=1000)


# Main function
def make_datasets():
    # Read file
    print('Reading input file...')
    df = pd.read_csv(INPUT_FILENAME, 
                    converters={'input_ids': lambda x: json.loads(x),
                                'attention_mask': lambda x: json.loads(x)}, 
                                sep='\t')


    # MAKE TOKENIZATION STEP HERE

    # MAKE DATASET CREATION FOR NEW PARADIGM HERE

    # GIVE EXAMPLES AN ID

    # THINK A BIT ABOUT HOW TO STORE

    # Create and save toy and full datasets
    print('Creating and saving toy dataset...')
    _process_and_save(df, gsplit_small, 'split_stats_small.png', 
                      dsname='nn1_small',dev=False)
    print('Creating and saving full dataset...')
    _process_and_save(df, gsplit_main, 'split_stats.png', 
                      dsname= 'nn1_full', dev=True)

    # Create and save shuffled dataset
    print('Creating and saving shuffled dataset')
    df_shuffled = df.copy()
    df_shuffled['author'] = np.random.permutation(df_shuffled['author'].values)
    _process_and_save(df, gsplit_main, 'split_stats_shuffled.png', 
                      dsname='nn1_shuffled', dev=True)


if __name__=='__main__':
    make_datasets()
