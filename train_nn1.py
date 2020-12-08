from tensorflow import keras
import tensorflow as tf
from keras.layers import Input
import glob
from tools.tfrecords import load_tfrecord_nn1
from models.nn1 import Model
from pathlib import Path

# Define folder to save model graph
MODEL_FILE = str(Path('figures') / 'nn1.png')

# Define parameters for model compilation
optimizer=keras.optimizers.Adam(learning_rate=5e-5)
loss='categorical_crossentropy'
metrics=['categorical_accuracy']

# Define inputs
input_ids = Input(shape=(None, 512), dtype='int32', name='input_ids')
attention_mask = Input(shape=(None, 512), dtype='int32', name='attention_mask')

#def train_nn1():
outs = Model()([input_ids, attention_mask])
model = keras.Model([input_ids, attention_mask], outs)
keras.utils.plot_model(model, MODEL_FILE, show_shapes=True)
model.summary()
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Grab dataset
fnames = glob.glob('datasets/test/*')
ds = load_tfrecord_nn1(filenames=fnames, compression_type='GZIP')

# Fit dataset
history = model.fit(ds.batch(1), epochs=2, verbose=1) 

    # TO DOS:
    # Fix batch size issue - probably need padded batches
    # Add logging of metrics (loss etc)
    # Add printing examples
    # Add prefetch
    # Add parallelization options

