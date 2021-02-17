import json
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
from IPython.display import clear_output
import itertools


def merge_csv(dir, sep='\t'):
    ''' Read in a list of files and return merged dataframe '''
    path = Path(dir)
    flist = os.listdir(dir)
    for idx, f in enumerate(flist):
        clear_output(wait=True)
        print(f'Reading file {idx+1} out of {len(flist)}')
        try:
            subdf = pd.read_csv(str(path/f), sep=sep)
            subdf = subdf.drop_duplicates(subset='selftext')
            if idx == 0:
                df = subdf.copy()
            else:
                df = pd.concat([df, subdf], ignore_index=True)
        except:
            print(f'Skipping {f} - error')
    return df


def add_aggregate_metrics(data, agg_by, agg_dict, colnames, agg_on='selftext', inplace=True):
    ''' Adds to df columns with aggregate metrics (e.g. nr posts per subreddit) '''
    aggdf = data[[agg_by, agg_on]].groupby(agg_by).aggregate(agg_dict).reset_index()
    aggdf.columns = colnames
    if inplace:
        if colnames[1] in data.columns:
            data = data.drop(colnames[1], axis=1)
        return pd.merge(data, aggdf)
    else:
        return aggdf


def update_aggregates(data, metrics='all'):
    if metrics == 'all':
        metrics = ['user_posts_count', 'subreddit_posts_count',
                   'subreddit_nr_unique_users', 'user_nr_unique_subreddits']
    for m in metrics:
        if m == 'user_posts_count':
            data = add_aggregate_metrics(data, 'author', 'count', 
                                         ['author', 'user_posts_count'])
        elif m == 'subreddit_posts_count':
            data = add_aggregate_metrics(data, 'subreddit', 'count',
                                         ['subreddit', 'subreddit_posts_count'])
        elif m == 'subreddit_nr_unique_users':
            data = add_aggregate_metrics(data, 'subreddit', lambda x: x.nunique(), 
                                         ['subreddit', 'subreddit_nr_unique_users'],
                                         agg_on='author')      
        elif m == 'user_nr_unique_subreddits':
            data = add_aggregate_metrics(data, 'author', lambda x: x.nunique(), 
                                         ['author', 'user_nr_unique_subreddits'],
                                         agg_on='subreddit')
            
    return data


def plot_aggregates(data, figsize=(12,15), vlines=None, 
                    colors=None, bins=None, log=None, nrows=4, 
                    ncols=1, **kwargs):
    ''' Plot all aggregate metrics '''
    mdict = {'user_posts_count': ['author', '# posts', '# users'], 
             'user_nr_unique_subreddits':['author', '# subreddits', '# users'],
             'subreddit_posts_count': ['subreddit', '# post', '# subreddits'], 
             'subreddit_nr_unique_users': ['subreddit', '# users', '# subreddits']}
    bins = bins or [50] * 4
    log = log or [True] * 4
    colors = colors or ['orange', 'darkorange', 'darkred', 'black']
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)
    ax_idx = list(itertools.product(range(nrows), range(ncols)))
    for idx, m in enumerate(mdict.keys()):
        if log[idx] == True:
            ax[ax_idx[idx]].set_xscale('log')
        if vlines:
            for i, v in enumerate(vlines):
                ax[ax_idx[idx]].axvline(v, linestyle='--', c=colors[i])
        sns.histplot(data.groupby(mdict[m][0]).aggregate({m: 'first'}), 
                     bins=bins[idx], ax=ax[ax_idx[idx]])
        ax[ax_idx[idx]].set_xlabel(mdict[m][1])
        ax[ax_idx[idx]].set_ylabel(mdict[m][2])
        ax[ax_idx[idx]].legend('')
    plt.tight_layout()
    plt.show()


def log_size(data, sdict, name, save_file=None):
    ''' Track dataset size over preprocessing '''
    sdict['names'].append(name)
    sdict['users'].append(data.author.nunique())
    sdict['posts'].append(data.shape[0])
    sdict['subreddits'].append(data.subreddit.nunique())
    if save_file is not None:
        json.dump(sdict, open(save_file, 'w'))
    return sdict