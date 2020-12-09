#!/usr/bin/env python3
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import Input
import glob
from tools.tfrecords import load_tfrecord_nn1
from models.nn1 import CustomBertLayer, TransformerBlock, ClassificationBlock
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

# Train
bert = CustomBertLayer(trainable=False)([input_ids, attention_mask])
transformer = TransformerBlock(name='transformer')(bert)
predictions = ClassificationBlock(name='one_hot_subreddit')(transformer)
model = keras.Model([input_ids, attention_mask], predictions)

# Visualize
keras.utils.plot_model(model, MODEL_FILE, show_shapes=True)
model.summary()
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Grab dataset
fnames = glob.glob('datasets/nn1_small/train/*')
ds = load_tfrecord_nn1(filenames=fnames, compression_type='GZIP')

# Fit dataset
history = model.fit(ds.batch(1), epochs=2, verbose=1) 

# TO DOS:
# Fix batch size issue - maybe no batching? 
# Add logging
# Add printing examples
# Add prefetch
# Add parallelization options

