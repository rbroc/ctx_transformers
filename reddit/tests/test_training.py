import tensorflow as tf
from reddit import Trainer, Logger
from reddit.losses import TripletLossBase
from reddit.models import BatchTransformer
from transformers import TFDistilBertModel
from official.nlp.optimization import create_optimizer
import pytest
from utils import build_distilbert_multiple_input
from reddit.utils import load_tfrecord, pad_and_stack
import shutil
import glob
from pathlib import Path
import json


WEIGHTS = 'distilbert-base-uncased'
TRAIN_VARS = ['losses','metrics', 'dist_pos', 'dist_neg', 'dist_anchor']
TEST_VARS = ['test_' + v for v in TRAIN_VARS]


def test_trainer_logging():
    loss = TripletLossBase(1)
    optimizer = create_optimizer(2e-5, num_train_steps=100, num_warmup_steps=10)
    model = BatchTransformer(TFDistilBertModel, WEIGHTS)
    trainer = Trainer(model, loss, optimizer, 
                      strategy=None, 
                      n_epochs=1, steps_per_epoch=100, 
                      log_every=2, 
                      train_vars=TRAIN_VARS, 
                      test_vars=TEST_VARS, 
                      log_path='tmp',
                      checkpoint_device=None,
                      distributed=False)
    model_path = Path(f'BatchTransformer-{WEIGHTS}') / 'triplet_loss_margin-1'
    assert all([v in trainer.logger.logdict.keys() for v in TRAIN_VARS+TEST_VARS])
    # Create some mock data
    mock_losses = tf.random.normal((5,))
    mock_metrics = tf.constant([1.,0.,1.,1.,0.])
    mock_pos_distances = tf.random.normal((5,))
    mock_neg_distances = tf.random.normal((5,))
    mock_anch_distances = tf.random.normal((5,))
    mock_ids = tf.constant([4,5,10,234,2])
    mock_test_losses = tf.random.normal((3,))
    mock_test_metrics = tf.constant([1.,1.,0.])
    mock_pos_distances_test = tf.random.normal((3,))
    mock_neg_distances_test = tf.random.normal((3,))
    mock_anch_distances_test = tf.random.normal((3,))
    mock_test_ids = tf.constant([8,11,77])
    for k in ['metrics', 'checkpoint', 'optimizer']:
        assert (Path('tmp') / k / model_path).exists()
    # Test correct logging of metrics
    for _ in range(2):
        trainer.logger.log(vars=[mock_losses, mock_metrics, mock_pos_distances, 
                                 mock_neg_distances, mock_anch_distances],
                                 epoch=1, example_ids=mock_ids, batch=1, train=True)
    assert all([len(v)==10 for k,v in trainer.logger.logdict.items() 
                           if k in TRAIN_VARS + ['example_ids']])
    for _ in range(2):
        trainer.logger.log(vars=[mock_test_losses, mock_test_metrics,
                                 mock_pos_distances_test, mock_neg_distances_test,
                                 mock_anch_distances_test],
                                 epoch=1, example_ids=mock_test_ids, 
                                 train=False)
    assert all([len(v)==6 for k,v in trainer.logger.logdict.items() 
                          if k in TEST_VARS + ['test_example_ids']])
    # Check if files are saved
    trainer.logger.log(vars=[mock_losses, mock_metrics, mock_pos_distances, 
                             mock_neg_distances, mock_anch_distances],
                             epoch=1, example_ids=mock_ids, batch=26, train=True)
    assert all([len(v)==15 for k,v in trainer.logger.logdict.items() 
                          if k in TRAIN_VARS + ['example_ids']])  
    target_path = Path('tmp') / 'metrics' / model_path / 'epoch-1' / 'log.json'
    assert (target_path).exists()
    with open(str(target_path)) as fh:
        d = json.load(fh)
    assert all([len(v)==15 for k,v in d.items() if k in TRAIN_VARS + ['example_ids']])
    assert all([len(v)==6 for k,v in d.items() if k in TEST_VARS + ['test_example_ids']])
    shutil.rmtree('tmp')


def test_trainer_checkpoints():
    loss = TripletLossBase(1)
    optimizer = create_optimizer(2e-5, num_train_steps=100, num_warmup_steps=10)
    model = BatchTransformer(TFDistilBertModel, WEIGHTS)
    trainer = Trainer(model, loss, optimizer, 
                      strategy=None, 
                      n_epochs=1, steps_per_epoch=100, 
                      log_every=2, 
                      train_vars=TRAIN_VARS, 
                      test_vars=TEST_VARS, 
                      log_path='tmp',
                      checkpoint_device=None,
                      distributed=False)
    model_path = Path(f'BatchTransformer-{WEIGHTS}') / 'triplet_loss_margin-1'
    try:
        trainer.model_ckpt.save(1, 22)
        trainer.opt_ckpt.save(1,22)
    except:
        IOError('Saving of checkpoints failed')
    target_path = Path('tmp') / 'checkpoint' / model_path / 'epoch-1'
    target_path1 = target_path / 'batch-22-of-100.data-00000-of-00001'
    target_path2 = target_path / 'batch-22-of-100.index'
    assert target_path1.exists()
    assert target_path2.exists()
    try:
        Trainer(model, loss, optimizer, steps_per_epoch=100, 
                strategy=None, n_epochs=1,
                start_epoch=2, log_path='tmp', 
                checkpoint_device=None)
    except:
        IOError('Loading checkpoints failed')
    shutil.rmtree('tmp')


def test_trainer_train():
    n_train_examples = 3
    n_test_examples = 5
    loss = TripletLossBase(1)
    optimizer = create_optimizer(2e-5, num_train_steps=100, num_warmup_steps=10)
    model = BatchTransformer(TFDistilBertModel, WEIGHTS)
    trainer = Trainer(model, loss, optimizer, 
                      strategy=None, 
                      n_epochs=1, 
                      steps_per_epoch=3, 
                      log_every=2, 
                      train_vars=TRAIN_VARS, 
                      test_vars=TEST_VARS, 
                      log_path='tmp',
                      checkpoint_device=None,
                      distributed=False)
    model_path = Path(f'BatchTransformer-{WEIGHTS}') / 'triplet_loss_margin-1'
    examples, _ = build_distilbert_multiple_input(weights=WEIGHTS, 
                                                  n_examples=n_train_examples)
    test_examples, _ = build_distilbert_multiple_input(weights=WEIGHTS,
                                                       kind='test',
                                                       n_padded=0, 
                                                       n_examples=n_test_examples)
    ds = tf.data.Dataset.from_tensor_slices(examples)
    ds_test = tf.data.Dataset.from_tensor_slices(test_examples)
    trainer.run(dataset_train=ds, dataset_test=ds_test)
    model_ckpt_path = Path('tmp') / 'checkpoint' / \
                      model_path / 'epoch-0' / 'batch-3-of-3.index'
    model_ckpt_path.exists()
    opt_ckpt_path = Path('tmp') / 'optimizer' / \
                      model_path / 'epoch-0' / 'batch-3-of-3.pkl'
    opt_ckpt_path.exists()
    metrics_path = Path('tmp') / 'metrics' / \
                      model_path / 'epoch-0' / 'log.json'
    with open(str(metrics_path)) as fh:
        d = json.load(fh)
    assert all([len(v) == n_train_examples 
                for v in d.values() if v in TRAIN_VARS])
    assert all([len(v) == n_test_examples 
                for v in d.values() if v in TEST_VARS])
    assert len(d['example_ids']) == n_train_examples * 2
    assert len(d['test_example_ids']) == n_test_examples * 2
    shutil.rmtree('tmp')

