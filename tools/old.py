import pandas as pd 
import numpy as np
import tensorflow as tf
from pathlib import Path
import os
import pickle as pkl
from IPython.display import clear_output
import abc

def save_dataset(ds, ds_name, type='train', ds_dir='datasets', save_spec=True):
    ''' Saves dataset in relevant folder (ds is a TF dataset) '''
    dpath = Path(ds_dir)
    outdir = dpath / ds_name
    savedir = outdir / type
    tf.data.experimental.save(ds, str(savedir))
    if save_spec:
        with open(str(outdir / 'spec.txt'), 'wb') as f:
            pkl.dump(ds.element_spec, f)


def load_datasets(ds_name, load='all', ds_dir='datasets'):
    ''' Loads train, val and test components of a dataset '''
    dpath = Path(ds_dir)
    outdir = dpath / ds_name
    spec = pkl.load(open(str(outdir / 'spec.txt'), 'rb'))
    dsets = []
    if load == 'all':
        load = ['train', 'val', 'test']
    if 'train' in load:
        dsets.append(tf.data.experimental.load(str(outdir / 'train'), 
                                               element_spec=spec))
    if 'val' in load:
        dsets.append(tf.data.experimental.load(str(outdir / 'val'), 
                                               element_spec=spec))
    if 'test' in load:
        dsets.append(tf.data.experimental.load(str(outdir / 'test'), 
                                               element_spec=spec))
    return tuple(dsets)
