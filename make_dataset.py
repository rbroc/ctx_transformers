import json
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path
from tools.datasets import (split_dataset, stack_examples, 
                            plot_split_stat, save_dataset, 
                            plot_subreddit_distribution)

PROCESSED_PATH = Path('processed') / 'pushshift'
LABEL_PATH = Path('labels')
FIG_PATH = Path('figures')
INPUT_FILENAME = PROCESSED_PATH / '500sr_encoded.txt'

# Initialize split utils
gsplit_main = GroupShuffleSplit(n_splits=1, train_size=.70, 
                                test_size=.15, random_state=0)
gsplit_small = GroupShuffleSplit(n_splits=1, train_size=.25, 
                                 test_size=.05, random_state=0)


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

    # Create and save toy datasets
    print('Creating and saving toy dataset...')
    train_small, test_small = split_dataset(df, gsplit_small, dev=False)
    plot_split_stat(train_small, test_small, 
                    save=True, fname=str(FIG_PATH/'split_stats_small.png'))
    train_small, test_small = stack_examples([train_small, test_small])
    for d in [train_small, test_small]:
        save_dataset(d, path, shard=True, filename=None,
                     compression='GZIP', n_shards=1000) # define path and 

    # Create and save full datasets
   print('Creating and saving full dataset...')
    train, test, dev = split_dataset(df, gsplit_main, dev=True)
    plot_split_stat(train, test, save=True, 
                    fname=str(FIG_PATH/'split_stats.png'))
    train, test, dev = stack_examples([train, test, dev])
    for d in [train, test, dev]:
        save_dataset(d, path, shard=True, filename=None,
                     compression='GZIP', n_shards=1000) # define path and 

    # Create and save shuffled datasets
    print('Creating and saving shuffled dataset...')
    df_shuffled = df.copy()
    df_shuffled['author'] = np.random.permutation(df_shuffled['author'].values)
    train, test, dev = split_dataset(df_shuffled, gsplit_main, dev=True)
    plot_split_stat(train, test, save=True, 
                    fname=str(FIG_PATH/'split_stats.png'))
    train, test, dev = stack_examples([train, test, dev])
    for d in [train, test, dev]:
        save_dataset(d, path, shard=True, filename=None,
                     compression='GZIP', n_shards=1000) # define path and 


if __name__=='__main__':
    make_datasets()
