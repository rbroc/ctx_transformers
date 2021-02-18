import tensorflow as tf
from pathlib import Path
import json


class Logger:
    def __init__(self, trainer):
        self.logdict = {}
        self.trainer = trainer
        self.outfolder = Path(f'{self.trainer.model.name}') / self.trainer.loss.name
        self.outfolder.mkdir(exist_ok=True, parents=True)
        self._reset()

    def _reset(self):
        for v in self.trainer.train_vars + self.trainer.test_vars:
            self.logdict[v] = []
    
    def log(self, data, epoch, n=None, train=True):
        for idx, d in enumerate(data):
            fv = [v.numpy() for v in d] 
            if train:
                self.logdict[self.trainer.train_vars[idx]] += fv
                if (n == self.trainer.examples_per_epoch) or \
                    (n % self.trainer.log_every == 0):
                    self._save(epoch)
            else:
                self.logdict[self.trainer.test_vars[idx]] += fv

    def _save(self, epoch):
        outfile = f'epoch-{str(epoch)}.json'
        outpath = str(self.outfolder / outfile)
        with open(outpath, 'w') as fh:
            fh.write(json.dumps(self.logdict))


class Checkpoint:
    def __init__(self, trainer):
        self.trainer = trainer
        # self.optimizer_folder
        # self.checkpoint_folder
        # check if they exist, otherwise create
        # Load if required (see belo)
        if self.trainer.start_epoch != 0:
            if self.trainer.load_checkpoints != False:
                self._load()

    
    def _load(self):
        # Load model/loss/epoch checkpoint
        # Load model/loss/epoch optimizer weights
        pass

    def dump(self):
        # Save model/loss/epoch checkpoint
        # Save model/loss/epoch optimizer weights
        pass
