from reddit.training import Trainer
from reddit.losses import TripletLossBase
from reddit.models import BatchTransformer
from reddit.utils import load_tfrecord, pad_and_stack
from transformers import TFDistilBertModel
from official.nlp.optimization import create_optimizer
import pytest
from pathlib import Path
import glob
import shutil
import json

WEIGHTS = 'distilbert-base-uncased'
TRAIN_VARS = ['losses','metrics', 'dist_pos', 'dist_neg', 'dist_anchor']
TEST_VARS = ['test_' + v for v in TRAIN_VARS]
ds_files = glob.glob(str(Path('data')/'sample_dataset*'))

def test_dataset_train():
    loss = TripletLossBase(1)
    optimizer = create_optimizer(2e-5, num_train_steps=100, num_warmup_steps=10)
    model = BatchTransformer(TFDistilBertModel, WEIGHTS)
    trainer = Trainer(model, loss, optimizer, 
                      strategy=None, 
                      n_epochs=1, 
                      steps_per_epoch=6, 
                      log_every=2,
                      train_vars=TRAIN_VARS, 
                      test_vars=TEST_VARS, 
                      log_path='tmp',
                      checkpoint_device=None,
                      distributed=False)
    model_path = Path(f'BatchTransformer-{WEIGHTS}') / 'triplet_loss_margin-1'
    # Process dataset
    ds = load_tfrecord(ds_files)
    ds_train = ds.take(6)
    ds_test = ds.skip(6).take(2)
    ds_train = pad_and_stack(ds_train, pad_to=[5,1,1]).batch(2)
    ds_test = pad_and_stack(ds_test, pad_to=[5,1,1]).batch(2)
    # Try training
    trainer.train(dataset_train=ds_train, dataset_test=ds_test)
    metrics_path = Path('tmp') / 'metrics' / \
                      model_path / 'epoch-0' / 'log.json'
    with open(str(metrics_path)) as fh:
        d = json.load(fh)
    assert all([len(v) == 6
                for v in d.values() if v in TRAIN_VARS])
    assert all([len(v) == 2
                for v in d.values() if v in TEST_VARS])
    assert len(d['example_ids']) == 6
    assert len(d['test_example_ids']) == 2
    shutil.rmtree('tmp')
