import tensorflow as tf


def filter_triplet_by_n_anchors(x, min_anchors):
    ''' Filtering function to remove stuff which is too short to be masked '''
    return tf.math.greater(x['n_anchor'], min_anchors-1)


def pad_and_stack_triplet(dataset, pad_to=[20,1,1], min_anchors=None):
    ''' Pads the dataset according to specified number of posts 
        passed via pad_to (anchor, positive, negative) and stacks
        negative, positive and anchor posts vertically.
        Shuffles the anchor comments
    Args:
        dataset (TFDataset): dataset to pad and stack 
        pad_to (list or tuple): list containing number of posts
            to pad to, i.e., [n_anchor_posts, n_positive_posts,
            n_negative_posts]
    '''
    if min_anchors:
        dataset = dataset.filter(lambda x: filter_triplet_by_n_anchors(x, min_anchors))
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


def _get_feature_names(is_context):
    ''' Returns feature names for mlm dataset '''
    if is_context:
        return (['masked_target_iids', 'context_iids'], 
                ['target_amask', 'context_amask'])
    else:
        return (['masked_target_iids'], 
                ['target_amask'])

    
def _get_mask(target, mask_token_id, mask_proportion):
    ''' Given a tensor of input ids, picks random tokens 
        to mask, and returns both index and token. 
        It masks mask_proportion of the input '''
    n_tokens = tf.math.count_nonzero(target)
    n_masked = tf.cast((tf.cast(n_tokens, 
                                tf.float32) - 2.0) * mask_proportion, 
                       tf.int32) 
    mask_idxs = tf.cast(tf.random.shuffle(tf.range(1, n_tokens-1))[:n_masked],
                        tf.int32)
    bool_mask = tf.reduce_any(tf.equal(tf.expand_dims(tf.range(0,512),0),
                                       tf.expand_dims(mask_idxs, -1)), 
                              axis=0,
                              keepdims=True)
    label_mask = tf.where(bool_mask, target, 0)
    masked_target = tf.where(bool_mask, mask_token_id, target)
    return masked_target, label_mask, mask_idxs
    

def _update_and_return(d, newdict):
    ''' Util function (updates dictionary and returns it)'''
    d.update(newdict)
    return d


def remove_short_targets(x, mask_proportion):
    ''' Filtering function to remove stuff which is too short to be masked '''
    return tf.not_equal(tf.math.floor((tf.cast(tf.math.count_nonzero(x['target_iids']), 
                                              tf.float32) - 2.0) * mask_proportion), 
                        0.0)

    
def mask_and_stack_mlm(dataset, is_context=True, mask_proportion=.15):
    ''' Masks a random item in the target tensor (but could be more, 
        see docstring for _get_mask) for each example, stacks 
        target and context (if context is given), returns dataset 
        with input_ids, attention_mask, id, mask_idx, and mask_token
    '''
    iid_feat, mask_feat = _get_feature_names(is_context)
    dataset = dataset.filter(lambda x: remove_short_targets(x, mask_proportion))
    dataset = dataset.map(lambda x: _update_and_return(x, 
                                                       dict(zip(['masked_target_iids', 
                                                                 'labels'],
                                                                 _get_mask(x['target_iids'],
                                                                           x['mask_token_id'],
                                                                           mask_proportion)
                                                               ))))
    dataset = dataset.map(lambda x: {'input_ids': tf.concat(values=[x[f] 
                                                                    for f in iid_feat],
                                                            axis=0),
                                     'attention_mask': tf.concat(values=[x[f] 
                                                                         for f in mask_feat],
                                                                 axis=0),
                                     'id': x['example_id'],
                                     'labels': x['labels']})
    return dataset
    

def prepare_agg(dataset):  
    dataset = dataset.map(lambda x: {'input_ids': x['iids'],
                                     'attention_mask': x['amask'],    
                                     'id': x['author_id'],
                                     'avg_score': x['avg_score'],
                                     'avg_comm': x['avg_comm'], 
                                     'avg_posts': x['avg_posts']})
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
        tuning, training, valdiation and test set
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
