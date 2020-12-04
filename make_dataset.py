import json
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path
from tools.datasets import (split_dataset, stack_examples, 
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

    # One-hot encoding of subreddits
    print('One-hot encoding subreddits...')
    df = df.sort_values('created_utc').reset_index(drop=True)
    df['one_hot_subreddit'] = pd.get_dummies(df['subreddit']).values.tolist()
    df['label_subreddit'] = df['one_hot_subreddit'].apply(lambda x: np.where(np.array(x) == 1)[0][0])

    # Store label mapping
    print('Saving label mappings...')
    label_map = df[['subreddit', 'one_hot_subreddit']]\
                .groupby('subreddit')\
                .aggregate('first')
    label_map['one_hot_subreddit'] = [np.where(np.array(l) == 1)[0][0]
                                      for l in label_map['one_hot_subreddit']]
    label_dict = dict(zip(list(label_map['one_hot_subreddit']), 
                          list(label_map.index)))
    json.dump(label_dict, open(str(LABEL_PATH / 'reddit_2008.json'), 'w'))

    # Plot distribution over whole dataset
    print('Saving subreddit distribution...')
    plot_subreddit_distribution(d=df, 
                                save=True, 
                                fname=str(FIG_PATH/'subreddit_distribution_nn1.png'))

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
