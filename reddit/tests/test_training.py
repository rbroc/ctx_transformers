import tensorflow as tf
from reddit import Trainer, Logger
from reddit.losses import TripletLossBase
from reddit.models import BatchTransformer
from transformers import TFDistilBertModel
from official.nlp.optimization import create_optimizer
import pytest
from pathlib import Path
import json

def test_trainer():
    loss = TripletLossBase(1)
    optimizer = create_optimizer(2e-5, num_train_steps=100, num_warmup_steps=10)
    model = BatchTransformer(TFDistilBertModel, 'distilbert-base-uncased')
    train_vars = ['losses','metrics', 'distance_pos', 'distance_neg', 'distance_anchor']
    test_vars = ['test_losses', 'test_metrics']
    trainer = Trainer(model, loss, optimizer, strategy=None, n_epochs=1,
                      steps_per_epoch=100, log_every=2, 
                      train_vars=train_vars, test_vars=test_vars, log_path='tmp')
    assert all([v in trainer.logger.logdict.keys() for v in train_vars+test_vars])

    mock_losses = tf.random.normal((5,))
    mock_metrics = tf.constant([1.,0.,1.,1.,0.])
    mock_pos_distances = tf.random.normal((5,))
    mock_neg_distances = tf.random.normal((5,))
    mock_anch_distances = tf.random.normal((5,))
    mock_ids = tf.constant([4,5,10,234,2])
    mock_test_losses = tf.random.normal((3,))
    mock_test_metrics = tf.constant([1.,1.,0.])
    mock_test_ids = tf.constant([8,11,77])
    model_path = Path('BatchTransformer-distilbert-base-uncased') / 'triplet_loss_margin-1'
    for k in ['metrics', 'checkpoint', 'optimizer']:
        assert (Path('tmp') / k / model_path).exists()
    # Test correct logging of metrics
    for _ in range(2):
        trainer.logger.log(vars=[mock_losses, mock_metrics, mock_pos_distances, 
                                 mock_neg_distances, mock_anch_distances],
                                epoch=1, example_ids=mock_ids, batch=1, train=True)
    assert all([len(v)==10 for k,v in trainer.logger.logdict.items() 
                           if k in train_vars + ['example_ids']])
    for _ in range(2):
        trainer.logger.log(vars=[mock_test_losses, mock_test_metrics],
                                 epoch=1, example_ids=mock_test_ids, train=False)
    assert all([len(v)==6 for k,v in trainer.logger.logdict.items() 
                          if k in test_vars + ['test_example_ids']])
    # Check if files are saved
    trainer.logger.log(vars=[mock_losses, mock_metrics, mock_pos_distances, 
                             mock_neg_distances, mock_anch_distances],
                             epoch=1, example_ids=mock_ids, batch=26, train=True)
    assert all([len(v)==15 for k,v in trainer.logger.logdict.items() 
                          if k in train_vars + ['example_ids']])  
    target_path = Path('tmp') / 'metrics' / model_path / 'epoch-1' / 'log.json'
    assert (target_path).exists()
    with open(str(target_path)) as fh:
        d = json.load(fh)
    assert all([len(v)==15 for k,v in d.items() if k in train_vars + ['example_ids']])
    assert all([len(v)==6 for k,v in d.items() if k in test_vars + ['test_example_ids']])
    
    # test checkpoint model (content, filename, frequency)
    # test checkpoint optimizer (content, filename, frequency)
    # check train step and test step
    # global test (notebook)