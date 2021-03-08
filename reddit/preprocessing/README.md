### Scripts for dataset creation
The scripts contained in this folder cover all the steps needed to create datasets of Reddit posts in the format required by the models.
Dataset creation steps and the corresponding scripts are the following:
- ```download.py```: downloads Reddit submissions from the Pushshift dump at https://files.pushshift.io/reddit/submissions/, removes empty posts, crossposts, posts flagged as over_17 or posts with missing essential info, and stores into tsv files with up to 100000 posts each
- ```filter.py```: takes the outputs of ```download.py```, removes duplicates and non-English posts and only keeps posts from users with >5 posts, who contributed to >5 unique subreddits
- ```make_triplets.py```: for each user, picks one post to be used as positive example, and one post used as negative example for other users in a triplet loss learning framework. Also pairs up negative examples with respective users, and stores info on each example (id, user, anchor / positive / negative examples, + metadata) in a json file for easy inspection
- ```tokenize.py```: tokenizes all examples with a tokenizer of interest (defaults to ```distilbert-base-uncased```), and saves encoded posts
- ```make_tf_dataset.py```: stacks anchor, positive and negative example for each users, and saves the resulting dataset as TFRecord.
All outputs are saved in the ```data``` folder of the package.s