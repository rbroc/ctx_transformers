import tensorflow as tf


def pad_and_stack(dataset, pad_to=[20,1,1]):
    ''' Pads the dataset according to specified number of posts 
        passed via pad_to (anchor, positive, negative) and stacks
        negative, positive and anchor posts vertically
    Args:
        dataset (TFDataset): dataset to pad and stack 
        pad_to (list or tuple): list containing number of posts
            to pad to, i.e., [n_anchor_posts, n_positive_posts,
            n_negative_posts]
    '''
    dataset = dataset.map(lambda x: {'iids': x['iids'][:pad_to[0],:], 
                                     'amask': x['amask'][:pad_to[0],:],
                                     'pos_iids': x['pos_iids'][:pad_to[1],:],
                                     'pos_amask': x['pos_amask'][:pad_to[1],:],
                                     'neg_iids': x['neg_iids'][:pad_to[2],:],
                                     'neg_amask': x['neg_amask'][:pad_to[2],:],
                                     'author_id': x['author_id']})
    dataset = dataset.map(lambda x: {'iids': tf.pad(x['iids'], 
                                                    [[0,pad_to[0]-tf.shape(x['iids'])[0]],
                                                     [0,0]]),
                                     'amask': tf.pad(x['amask'], 
                                                     [[0,pad_to[0]-tf.shape(x['amask'])[0]],
                                                      [0,0]]), 
                                     'pos_iids': tf.pad(x['pos_iids'], 
                                                        [[0,pad_to[1]-tf.shape(x['pos_iids'])[0]],
                                                         [0,0]]),
                                     'pos_amask': tf.pad(x['pos_amask'], 
                                                         [[0,pad_to[1]-tf.shape(x['pos_amask'])[0]],
                                                          [0,0]]), 
                                     'neg_iids': tf.pad(x['neg_iids'], 
                                                        [[0,pad_to[2]-tf.shape(x['neg_iids'])[0]],
                                                         [0,0]]),
                                     'neg_amask': tf.pad(x['neg_amask'], 
                                                         [[0,pad_to[2]-tf.shape(x['neg_amask'])[0]],
                                                          [0,0]]),
                                     'author_id': x['author_id']})
    dataset = dataset.map(lambda x: {'input_ids': tf.concat([x['neg_iids'],
                                                             x['pos_iids'],
                                                             x['iids']], axis=0),
                                     'attention_mask': tf.concat([x['neg_amask'],
                                                                  x['pos_amask'],
                                                                  x['amask']], axis=0),
                                     'id': x['author_id']})
    return dataset
    

def split_dataset(dataset, size=None,
                  perc_train=.7, perc_val=.1, 
                  perc_test=.1, tuning=None):
    ''' Split dataset into training, validation and test set 
    Args:
        dataset (TFDataset): dataset to split (preprocessed and batched)
        size (int): number of examples from dataset. If None,
            the total number of examples is calculated and all examples 
            are used.
        perc_train (float): percentage of examples in training set
        perc_val (float): percentage of examples in training set
        perc_test (float): percentage of examples in test set
        tuning (optional): if provided, defines number of example for 
            additional tuning dataset.
    Returns:
        tuning, training, valdiation and test set (not distributed yet)
    ''' 
    if size is None:
        size = 0
        for _ in dataset:
            size += 1
        print(f'Number of total examples: {size}')
    size_train = int(size * perc_train)
    size_val = int(size * perc_val)
    size_test = int(size * perc_test)
    d_train = dataset.take(size_train)
    d_val = dataset.skip(size_train).take(size_val)
    d_test = dataset.skip(size_train + size_val).take(size_test)
    if tuning is None:
        return d_train, d_val, d_test
    else:
        d_tuning = dataset.take(tuning)
        return d_tuning, d_train, d_val, d_test
