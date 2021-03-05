import numpy as np
import pandas as pd
import timeit

def _tknz(x, tokenizer, n_wds):
    ''' Util function to tokenize text '''
    out = tokenizer.encode_plus(' '.join(x.split()[:n_wds]),
                                truncation=True, 
                                padding='max_length')
    return out

def tokenize(df, tknzr, n_wds=400, batch_size=100000):
    ''' Encodes all posts in the dataset using a given 
        huggingface tokenizer
    Args:
        df (pd.DataFrame): dataframe
        tokenizer (Tokenizer): huggingface tokenizer object
        n_wds (int): number of words to truncate input at 
            (for efficiency,truncation would happen anyway
            at token level)
        batch_size (int): how many posts to encode at a 
            time
    '''
    pid = list(np.arange(0, df.shape[0], batch_size)) 
    pid = pid.append(df.shape[0])
    start_t = timeit.default_timer() 
    current_t = start_t
    for i in range(len(pid) - 1):
        print(f'Timestamp previous step {current_t - start_t}')
        print(f'Encoding {pid[i]} to {pid[i+1]} of {df.shape[0]}')
        tknzd = df['selftext'][pid[i]:
                               pid[i+1]].apply(lambda x: _tknz(x, 
                                                               tknzr, 
                                                               n_wds))
        tknzd = pd.DataFrame(tknzd)
        if i == 0:
            alltkn = tknzd
        else:
            alltkn = pd.concat([alltkn, 
                                tknzd], 
                                ignore_index=True)
        current_t = timeit.default_timer()
    for c in ['input_ids', 'attention_mask']:
        df[c] = alltkn['selftext'].apply(lambda x: x[c])
    return df