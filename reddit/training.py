import tensorflow as tf
from tensorflow.keras.utils import Progbar
import json
from pathlib import Path
import numpy as np
from reddit import Logger

# Next
# Check loading, creation, etc
# Add model and optimizer logger

class Trainer:
    ''' Trainer class 
    Args:
        model: 
        loss_object: loss object
        optimizer: optimizer object
        strategy (tf.Strategy): distribution strategy 
    '''
    def __init__(self, model, 
                 loss_object, optimizer, strategy, 
                 n_epochs,
                 examples_per_epoch,
                 checkpoint_every=None,
                 log_every=100,
                 start_epoch=0):
        self.model = model
        self.loss_object = loss_object
        self.optimizer = optimizer
        self.strategy = strategy
        self.n_epochs = n_epochs
        self.start_epoch = start_epoch
        self.examples_per_epoch = examples_per_epoch
        self.checkpoint_every = checkpoint_every or examples_per_epoch
        train_vars = ['losses', 'metrics', 
                      'd_pos', 'd_neg', 'd_anch']
        test_vars = ['test_losses', 'test_metrics']
        self.logger = Logger(train_vars, test_vars, log_every)


    def _train_step(self, batch_in_replica):
        with tf.GradientTape() as tape:
            encodings, n_posts = self.model(batch_in_replica)
            l, m, d_pos, d_neg, d_anch = self.loss_object(encodings, 
                                                          n_posts)
        gradients = tape.gradient(l, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,
                                           self.model.trainable_variables))
        return l, m, d_pos, d_neg, d_anch

    def _test_step(self, batch_in_replica):
        encoding, n_posts = self.model(batch_in_replica)
        l, m = self.loss_object(encoding, n_posts)[:2]
        return l, m

    @tf.function
    def _run_train_step(self, global_batch):
        pr_l, pr_m, \
        pr_d_pos, pr_d_neg, pr_d_anch = self.strategy.run(self._train_step,
                                                          args=(global_batch,))
        ls, ms = pr_l.values, pr_m.values
        d_poss = pr_d_pos.values
        d_negs = pr_d_neg.values
        d_anchs = pr_d_anch.values
        return ls, ms, d_poss, d_negs, d_anchs

    @tf.function
    def _run_test_step(self, global_batch):
        pr_l, pr_m = self.strategy.run(self._test_step, args=(global_batch,))
        ls, ms = pr_l.values, pr_m.values
        return ls, ms

    def _run_train_epoch(self, epoch, dataset_train):
        print(f'Epoch {epoch+1} of {self.n_epochs}')
        pb = Progbar(self.examples_per_epoch, 
                     stateful_metrics=['loss', 'correct'])
        for n, example in enumerate(dataset_train):
            outs = self._run_train_step(example)
            self.logger.log(list(outs), epoch, n)
            loss, metric = tf.reduce_mean(outs[0]), tf.reduce_mean(outs[1])
            pb.add(1, values=[('loss', loss), ('correct', metric)])
        avg_loss = np.mean(self.logger.logdict['losses'])
        avg_metric = np.mean(self.logger.logdict['metrics'])
        print(f'Mean loss: {avg_loss}; '
              f'Mean metric: {avg_metric}')


    def _run_test_epoch(self, epoch, dataset_val):
        for example in dataset_val:
            outs = self._run_test_step(example)
            self.logger.log(list(outs), epoch, train=False)
        self.logger._save(epoch)
        avg_loss = np.mean(self.logger.logdict['test_losses'])
        avg_metric = np.mean(self.logger.logdict['test_metrics'])
        print(f'Mean validation loss: {avg_loss}; '
              f'Mean validation metric: {avg_metric}')


    def train(self, dataset_train, dataset_val):
        for epoch in range(self.start_epoch, 
                           self.n_epochs):
            shuffled = dataset_train.shuffle(self.examples_per_epoch)
            distributed = self.strategy.experimental_distribute_dataset(shuffled) 
            self._run_train_epoch(epoch, distributed)
            self._run_test_epoch(epoch, dataset_val)
            self.logger._reset()


''' 
    for epoch in range(int(load)+1,epochs):

      ckpt_epoch_dir = ckpt_dir / f'epoch_{epoch}'
      opt_epoch_dir = optimizer_dir / f'epoch_{epoch}'
      pb_i = Progbar(examples, stateful_metrics=['loss', 'correct'])

      

      for x in dataset_train:
        if (batch % 5000 == 0):
        log_weights(ckpt_epoch_dir, batch, model)
        log_optimizer(opt_epoch_dir, batch, optimizer)

      log_weights(ckpt_epoch_dir, batch, model)
      log_optimizer(opt_epoch_dir, batch, optimizer)



# Overall loop: 
# Define model
# Define loss
# Define optimizer
# Initialize trainer
# Run! 

  with strategy.scope():

    optimizer = create_optimizer(lr, 
                                 num_train_steps=tot_train_steps,
                                 num_warmup_steps=warmup_steps)
    
    if load:
      latest = tf.train.latest_checkpoint(f'checkpoints_shuffle_02_10/triplet_loss_{margin}/epoch_{load}')
      print(f'Loading checkpoint at checkpoint/triplet_loss_{margin}/epoch_{load}...')
      model.load_weights(latest, options=lh_options)

      opt_wfile = f'optimizers_shuffle_02_10/triplet_loss_{margin}/epoch_{load}/batch_{batch_nr}-of-27113.pkl'
      opt_weights = pkl.load(file=open(opt_wfile, 'rb'))
      optimizer._create_all_weights(model.trainable_variables)
      optimizer.set_weights(opt_weights)


  def log_weights(edir, b, model):
    edir.mkdir(exist_ok=True, parents=True)
    ckpt_path = edir /  f'batch_{b}-of-{n_train}'
    model.save_weights(filepath=ckpt_path, options=lh_options)
  
  def log_optimizer(edir, b, optimizer):
    edir.mkdir(exist_ok=True, parents=True)
    opt_wpath = edir / f'batch_{b}-of-{n_train}.pkl'
    pkl.dump(file=open(opt_wpath, 'wb'), 
             obj=optimizer.get_weights())

'''