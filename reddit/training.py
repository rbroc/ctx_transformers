import tensorflow as tf
from tensorflow.keras.utils import Progbar
import json
from pathlib import Path
import numpy as np
from reddit import (Logger, ModelCheckpoint,
                    OptimizerCheckpoint)
                    

class Trainer:
    ''' Trainer 
    Args:
        model: model object
        loss_object: loss object
        optimizer: optimizer object
        strategy (tf.Strategy): distribution strategy 
        n_epochs (int): number of training epochs
        steps_per_epoch (int): number of training steps/batches
        checkpoint_every (int): how often (in examples) model 
            and optimizer weights should be saved
        log_every (int): how often (in examples) training/test
            variables (loss, metrics, etc.) should be logged
        start_epoch (int): which epoch to start training from
        no_load (bool): if start_epoch > 0, defines whether 
            model and optimizer weights should be loaded.
            Default is False (model/optimizer weights loaded)
        train_vars (list): name of outputs from model to be 
            logged at training. Should have same order as model 
            outputs and contain at least 'losses' and 'metrics'
            If None, set to ['losses', 'metrics'].
        test_vars (list): name of outputs from model to be
            logged at training. Should have the same order as model
            outputs and contain at least 'test_losses' and 
            'test_metrics'. If None, set to ['test_losses', 
            'test_metrics'].
        checkpoint_device (str): argument to CheckpointOptions
        log_path (str): argument to Path, defines path where 
            metrics and checkpoints folder for logging are 
            located (created if not existing)
    '''
    def __init__(self, model, 
                 loss_object, optimizer, strategy, 
                 n_epochs, steps_per_epoch,
                 checkpoint_every=None, log_every=100,
                 start_epoch=0, no_load=False,
                 train_vars=None, test_vars=None,
                 checkpoint_device='/job:localhost',
                 log_path='..'):
        
        if train_vars:
            if (not isinstance(train_vars, list)) or (len(train_vars) < 2):
                raise ValueError('train_vars should have at least two '
                                  'elements (loss, metric)')
        if test_vars:
            if (not isinstance(test_vars, list)) or (len(test_vars) < 2):
                raise ValueError('train_vars should have at least two '
                                  'elements (test loss, test metric)')
        self.train_vars = train_vars or ['losses', 'metrics']
        self.test_vars = test_vars or ['test_losses', 'test_metrics']
        self.model = model
        self.loss_object = loss_object
        self.optimizer = optimizer
        self.strategy = strategy
        self.n_epochs = n_epochs
        self.steps_per_epoch = steps_per_epoch
        self.checkpoint_every = checkpoint_every or steps_per_epoch
        self.log_every = log_every
        self.start_epoch = start_epoch
        if (start_epoch > 0) and (no_load is False):
            self.load_epoch = start_epoch - 1
        else:
            self.load_epoch = None
        self.logger = Logger(self, log_path)
        self.model_ckpt = ModelCheckpoint(self, checkpoint_device, log_path)
        self.opt_ckpt = OptimizerCheckpoint(self, log_path)


    def _train_step(self, batch_in_replica):
        ''' Define training step (single replica) '''
        with tf.GradientTape() as tape:
            model_out = self.model(batch_in_replica)
            loss_out = self.loss_object(*model_out)
        gradients = tape.gradient(loss_out[0], self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,
                                           self.model.trainable_variables))
        return loss_out


    def _test_step(self, batch_in_replica):
        ''' Define test step (single replica) '''
        model_out = self.model(batch_in_replica)
        test_loss_out = self.loss_object(*model_out)[:2]
        return test_loss_out


    @tf.function
    def _run_distributed_step(self, global_batch, train=True):
        ''' Run training/test step on all replicas '''
        fn = self._train_step if train else self._test_step
        step_outs = self.strategy.run(fn, args=(global_batch,))
        return [getattr(o,'values') for o in step_outs]


    def _run_train_epoch(self, epoch, dataset_train):
        ''' Run one training epoch 
        Args:
            epoch (int): epoch number
            dataset_train (DistributedDataset): training set
        '''
        pb = Progbar(self.steps_per_epoch, 
                     stateful_metrics=['loss', 'correct'])

        for n, example in enumerate(dataset_train):
            outs = self._run_distributed_step(example)
            self.logger.log(list(outs), epoch, example['id'].values, n+1)
            loss, metric = tf.reduce_mean(outs[0]).numpy(), \
                            tf.reduce_mean(outs[1]).numpy()
            pb.add(1, values=[('loss', loss), ('correct', metric)])

            if ((n+1) % self.checkpoint_every == 0) or \
                (n+1 == self.steps_per_epoch):
                self.model_ckpt.save(epoch, n+1)
                self.opt_ckpt.save(epoch, n+1)

        avg_loss = tf.reduce_mean(self.logger.logdict['losses']).numpy()
        avg_metric = tf.reduce_mean(self.logger.logdict['metrics']).numpy()
        print(f'Mean loss: {avg_loss}; Mean metric: {avg_metric}')


    def _run_test_epoch(self, epoch, dataset_test):
        ''' Run one validation/test epoch 
        Args:
            epoch (int): epoch number 
            dataset_val (DistributedDataset): validation/test set 
        '''
        for example in dataset_test:
            outs = self._run_distributed_step(example, train=False)
            self.logger.log(list(outs), epoch, example['id'].values,
                            train=False)

        self.logger._save(epoch)
        avg_loss = tf.reduce_mean(self.logger.logdict['test_losses']).numpy()
        avg_metric = tf.reduce_mean(self.logger.logdict['test_metrics']).numpy()
        print(f'Mean test loss: {avg_loss}; Mean test metric: {avg_metric}')


    def train(self, dataset_train, dataset_test, shuffle=True):
        ''' Run full training 
        Args:
            dataset_train (Dataset): training set (not distributed)
            dataset_val (DistributedDataset): validation set    
        '''
        for epoch in range(self.start_epoch, self.n_epochs):
            if shuffle:
                dataset = dataset_train.shuffle(self.steps_per_epoch)
            else:
                dataset = dataset_train
            distributed = self.strategy.experimental_distribute_dataset(dataset) 
            print(f'Epoch {epoch+1}/{self.n_epochs}')
            self._run_train_epoch(epoch, distributed)
            self._run_test_epoch(epoch, dataset_test)
            self.logger._reset()