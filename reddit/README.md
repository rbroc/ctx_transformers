### Training context-aware encoders from Reddit data 
This package contains code to train context-aware language models (see README at root)
It contains:
- Scripts for download and preprocessing of Reddit posts (```preprocessing```) and a data folder (```data```)
- Classes to create BERT-like models with those data as input (```models.py```)
- A bunch of custom layers (```layers.py```)
- Custom loss classes (```losses.py```)
- Helper classes to manage (distributed or local) model training (```training.py```)
- Helper classes to log metrics and checkpoints during training (```logging.py```)
- Pytest-friendly tests for models, losses and training classes (```tests```)
- Some useful functions: ```utils```
