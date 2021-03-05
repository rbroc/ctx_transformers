import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from tools.tfrecords import save_tfrecord_nn1
from tools.preprocess import update_aggregates

# Stack examples
def _stack_fn(x):
    return [np.vstack(np.array(x))]

def stack_examples(datasets):
    stacked_ds = []
    for d in datasets:
        stacked_ds.append(d[ds_cols].groupby('author').aggregate(_stack_fn).reset_index())
    return stacked_ds


# IO
def _nn1_gen(ds):
    for i in range(ds.shape[0]):
        yield tuple( [ds[inpn].iloc[i][0] for inpn in input_names] ), \
              ds[output_names[0]].iloc[i][0]


def save_dataset(ds, path, **kwargs):
    ds = tf.data.Dataset.from_generator(generator=lambda: _nn1_gen(ds), 
                                        output_types=types)
    save_tfrecord_nn1(ds, path, **kwargs)
    return ds
    
