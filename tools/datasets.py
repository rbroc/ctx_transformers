
from tools.preprocess import update_aggregates
import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from tools.tfrecords import save_tfrecord_nn1


# Define useful variables for train/test split
x_cols = ['input_ids', 'attention_mask']
group_col = 'author'
y_col = 'label_subreddit'
ds_cols = ['author', 'input_ids', 'attention_mask', 'one_hot_subreddit']

# Define useful variables for tensorflow dataset for NN1
input_names = ['input_ids', 'attention_mask']
output_names = ['one_hot_subreddit']
types = (tf.int32, tf.int32), (tf.int32)


def plot_subreddit_distribution(d, save=False, fname=None):
    fig, ax = plt.subplots(figsize=(25,3))
    ax.axvline(250, linestyle='--', color='grey')
    sns.barplot(x=d['label_subreddit'].value_counts().index, \
                y=f['label_subreddit'].value_counts().values, \
                color='darkorange')
    plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    if save:
        plt.savefig(fname)
    else:
        plt.show()


def split_dataset(df, splitobj, dev=False):

    splits = list(splitobj.split(X=df[x_cols].values, 
                                    y=df[y_col].values, 
                                    groups=df[group_col].values))
    train_idx, test_idx = list(splits[0][0]), list(splits[0][1])
    train = update_aggregates(df.iloc[train_idx])
    test = update_aggregates(df.iloc[test_idx])
    print(f'There are {train.shape[0]} examples, \
            {train.author.nunique()} unique users in the training set')
    print(f'There are {test.shape[0]} examples, \
            {test.author.nunique()} unique users in the test set')
    if dev:
        dev = df.iloc[dev_idx]
        dev_idx = update_aggregates(list(set(range(df.shape[0])) - 
                                         set(train_idx + test_idx)))
        print(f'There are {dev.shape[0]} examples, \
             {dev.author.nunique()} unique users in the validation set')
        return train, test, dev
    else:
        return train, test


def plot_split_stat(train, test, dev=None, save=False, fname=None):
    ncol = 3 if dev else 2
    fig, ax = plt.subplots(nrows=1, ncols=ncol, figsize=(10,2), sharex=True)
    for a in ax:
        a.set_xscale('log')
        a.set_xlabel('# users')
        a.set_ylabel('# posts')
    sns.histplot(train[['author','user_posts_count']].groupby('author').aggregate('first'), \
                 ax=ax[0], bins=50, legend=None)
    ax[0].set_title('train')
    sns.histplot(test[['author','user_posts_count']].groupby('author').aggregate('first'), \
                 ax=ax[1], bins=50, legend=None)
    ax[1].set_title('test')
    ax[1].set_ylabel('')
    if dev:
        sns.histplot(dev[['author','user_posts_count']].groupby('author').aggregate('first'), \
                     ax=ax[2], bins=50, legend=None)
        ax[2].set_title('dev')
        ax[2].set_ylabel('')
    plt.tight_layout()
    if fname:
        plt.show()
    else:
        plt.savefig(fname)


def _stack_fn(x):
    return [np.vstack(np.array(x))]


def stack_examples(datasets):
    stacked_ds = []
    for d in datasets:
        stacked_ds.append(d[ds_cols].groupby('author').aggregate(_stack_fn).reset_index())
    return stacked_ds


def _nn1_gen(ds):
    for i in range(ds.shape[0]):
        yield tuple( [ds[inpn].iloc[i][0] for inpn in input_names] ), \
              ds[output_names[0]].iloc[i][0]


def save_dataset(ds, path, **kwargs):
    ds = tf.data.Dataset.from_generator(generator=lambda: _nn1_gen(ds), 
                                        output_types=types)
    save_tfrecord_nn1(ds, path, **kwargs)
    return ds
    
