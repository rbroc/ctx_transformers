import json
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import glob
import itertools


def read_files(dir, sep='\t', compression='gzip', **kwargs):
    ''' Read in a list of files and return merged dataframe 
    Args:
        dir (str): folder to read files from
        sep (str): csv file separator
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
    return df

