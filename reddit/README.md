### Training personality encoders from Reddit data
This package contains code to train language models fine-tuned to encode personality differences expressed in text from Reddit posts.
It contains:
- Scripts for download and preprocessing of Reddit posts (```preprocessing```) and a data folder (```data```)
- Classes to create BERT-like models with those data as input (```models.py```)
- Custom loss classes, e.g. triplet loss (```losses.py```)
- Helper classes to manage (distributed or local) model training (```training.py```)
- Helper classes to log metrics and checkpoints during training (```logging.py```)
- Pytest-friendly tests for models, losses and training classes (```tests```)
- Some useful functions: ```utils```