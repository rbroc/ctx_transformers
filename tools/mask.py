import pandas as pd 
import numpy as np
import random


def _compute_mask_perc(data, perclist):
    ''' Mask data in percentages defined by perclist '''
    cnt = int(data.author.nunique() / len(perclist))
    mask_cnt = []
    for i in range(len(perclist) - 1):
        mask_cnt += [perclist[i]] * cnt
    mask_cnt += [perclist[-1]] * (data.author.nunique() - (cnt*(len(perclist)-1)))
    random.seed(0)
    random.shuffle(mask_cnt)
    map_dict = dict(zip(data.author.unique(), mask_cnt))
    return map_dict


def _compute_mask_idx(x):
    ''' Compute mask ids for each user'''
    nr_posts = x['input_ids'][0].shape[0]
    nr_masks = int(nr_posts * x['mask_perc'])
    mask_idx = random.sample(range(nr_posts), nr_masks)
    return mask_idx


def _return_masked_encodings(x):
    ''' Extract encodings of masked items '''
    idx = x['mask_idx']
    return [x['input_ids'][0][idx]], [x['attention_mask'][0][idx]], [x['one_hot_subreddit'][0][idx]]


def _drop_masked(x):
    ''' Drop encodings from input (ids, attentions, truncated one-hot subreddits)'''
    idx = x['mask_idx']
    return ([np.delete(x['input_ids'][0], idx, axis=0)], 
            [np.delete(x['attention_mask'][0], idx, axis=0)], 
            [np.delete(x['one_hot_truncated'][0], idx, axis=0)])


def _explode_masked(x):
    nrows = x['input_ids_2'][0].shape[0]
    idx_cols = [c for c in x.index if '_2' not in c]
    s_inp = pd.DataFrame(zip(x['input_ids_2'][0], 
                             x['attention_mask_2'][0],
                             x['one_hot_subreddit_2'][0]), 
                             columns=[c for c in x.index if '_2' in c])
    x_dup = pd.concat([pd.DataFrame([x[idx_cols]], 
                      columns=idx_cols)] * nrows, 
                      ignore_index=True)
    for c in idx_cols:
        s_inp[c] = x_dup[c]
    s_inp = s_inp[x.index]
    return s_inp


def mask_dataset(data, perclist, drop_onehot=True, explode=True):
    stdf = data.copy()
    input_col = ['input_ids', 'attention_mask', 'one_hot_truncated']
    masked_col = ['input_ids_2', 'attention_mask_2', 'one_hot_subreddit_2']
    mdict = _compute_mask_perc(stdf, perclist)
    stdf['mask_perc'] = stdf.author.map(mdict)
    stdf[masked_col] = np.nan
    print('Computing masked index')
    stdf['mask_idx'] = stdf.apply(_compute_mask_idx, axis=1)
    print('Adding columns for masked items')
    stdf[masked_col] = stdf.apply(_return_masked_encodings, axis=1, result_type='expand')
    print('Dropping masked rows from input')
    stdf[input_col] = stdf.apply(_drop_masked, axis=1, result_type='expand')
    if drop_onehot:
        print('Dropping one-hot column for input')
        stdf = stdf.drop('one_hot_subreddit', axis=1)
    if explode:
        print('Exploding arrays into multiple examples')
        stdf = pd.concat(list(stdf.apply(_explode_masked, axis=1)))
    return stdf
