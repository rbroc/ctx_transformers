import json
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import glob
import itertools


def read_files(dir, sep='\t', drop_duplicates=True, 
               compression='gzip', **kwargs):
    ''' Read in a list of files and return merged dataframe 
    Args:
        dir (str): folder to read files from
        sep (str): csv file separator
        drop_duplicates (bool): whether duplicates should
            be removed (if so, drops duplicates every 5
            files)
        compression (str): type of compression of the input
            files
    '''
    flist = glob.glob(str(dir / '*'))
    for idx, f in enumerate(flist):
        print(f'Reading file {idx+1} out of {len(flist)}')
        try:
            subdf = pd.read_csv(f, sep=sep, 
                                compression=compression, **kwargs)
            if idx == 0:
                df = subdf
            else:
                df = pd.concat([df, subdf], ignore_index=True)
        except:
            print(f'Skipping {f} - error')
        if (idx % 5 == 0) and (drop_duplicates):
            df = df.drop_duplicates(subset=['author',
                                            'selftext'])
    return df


def compute_aggregates(df, group_by, agg_fn, 
                       colnames, target='selftext', 
                       inplace=True):
    ''' Computes aggregates of a variable
    Args:
        df (pd.DataFrame): dataframe
        group_by (str): column to group by
        agg_fn (dict, str or function): dictionary, str or 
            function specifying how to aggregate
        colnames (list): column names for aggregate dataframe 
            (grouping and target column)
        target (str): variables to compute aggregates on
        inplace (bool): if True, adds aggregates to dataframe, 
            else returns aggregate dataframe only
    '''
    aggdf = df[[group_by, target]].groupby(group_by)
    aggdf = aggdf.agg(agg_fn).reset_index()
    if colnames:
        aggdf.columns = colnames
    if inplace:
        if colnames[1] in df.columns:
            df = df.drop(colnames[1], axis=1)
        return pd.merge(df, aggdf)
    else:
        return aggdf


def update_aggregates(df, metrics='all'):
    '''  Update main aggregate metrics in dataframe
    Args:
        df (pd.DataFrame): dataframe
        metrics (str): one of 'n_user_posts',
            'n_user_subreddits', or 'all'
    '''
    if metrics == 'all':
        metrics = ['n_user_posts', 'n_user_subreddits']
    for m in metrics:
        if m == 'n_user_posts':
            data = compute_aggregates(df, 
                                      group_by='author', 
                                      agg_fn='count', 
                                      colnames=['author', 
                                                'n_user_posts'])
        elif m == 'n_user_subreddits':
            data = compute_aggregates(df, 
                                      group_by='author', 
                                      agg_fn=lambda x: x.nunique(), 
                                      colnames=['author', 
                                                'n_user_subreddits'],
                                      target='subreddit')
    return data


def plot_aggregates(df, figsize=(8,6),
                    bins=[50, 20, 50, 50], 
                    log=None, 
                    nrows=1, ncols=2, save_file=None, 
                    **kwargs):
    ''' Plot histograms of all aggregate metrics 
        (n_user_posts,  n_user_subreddits, 
        wn_subreddit_posts, n_subreddit_users)
    Args:
        df (pd.DataFrame): dataframe
        figsize (tuple): figure size
        bins (list): list of number of bins per plot
        log (bool): if x axis should be plotted in log scale
        nrows (int): number of rows in subplot
        ncols (int): number of cols in subplot
        save_file (str): if provided, saves the plot at specified
            file path
        kwargs: keyword arguments for plt.subplots
    '''
    mdict = {'n_user_posts': ['author', '# posts', '# users'], 
             'n_user_subreddits': ['author', '# subreddits', '# users']}
    bins = bins or [50] * 2
    log = log or [True] * 2
    _, ax = plt.subplots(nrows=nrows, ncols=ncols, 
                         figsize=figsize, **kwargs)
    ax_idx = list(itertools.product(range(nrows), 
                                    range(ncols)))
    for idx, m in enumerate(mdict.keys()):
        if log[idx] == True:
            ax[ax_idx[idx]].set_xscale('log')
        sns.histplot(df.groupby(mdict[m][0]).agg({m: 'first'}), 
                     bins=bins[idx], ax=ax[ax_idx[idx]])
        ax[ax_idx[idx]].set_xlabel(mdict[m][1])
        ax[ax_idx[idx]].set_ylabel(mdict[m][2])
        ax[ax_idx[idx]].legend('')
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file)
    else:
        plt.show()


def log_size(df, name, sdict=None, save_file=None):
    ''' Log (and save) dataset size at different stages
    Args:
        name (str): name of the preprocessing step
            (key to dictionary)
        sdict (dict): log dictionary (if None, creates
            an empty dictionary)
        save_file (str): if provided, saves the 
            dictionary at specified path as json
    '''
    if sdict is None:
        sdict = {}
        for k in ['names', 'users', 
                  'posts','subreddits']:
            sdict[k] = []
    sdict['names'].append(name)
    sdict['users'].append(len(set(df['author'].tolist())))
    sdict['posts'].append(df.shape[0])
    sdict['subreddits'].append(len(set(df['subreddit'].tolist())))
    if save_file is not None:
        json.dump(sdict, open(save_file, 'w'))
    return sdict


def plot_size_log(sdict, save_file=None):
    ''' Plots change in dataset metrics (size, n_posts, etc)
        over preprocessing steps
    Args:
        sdict (dict): log dictionary
        save_file (str): if specified, saves figure to path
    '''
    _, ax = plt.subplots(ncols=3, figsize=(10,4), sharex=True)
    for idx, m in enumerate(['users', 'posts', 'subreddits']):
        sns.lineplot(x=sdict['names'], 
                     y=sdict[m], 
                     ax=ax[idx])
        ax[idx].set_ylabel(m)
        ax[idx].set_yscale('log')
        ax[idx].set_xticklabels(sdict['names'], 
                                rotation=90)
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file)
    else:
        plt.show()


