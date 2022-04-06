from reddit.utils import (pad_and_stack_triplet,
                          stack_classification,
                          mask_and_stack_mlm,
                          prepare_agg, prepare_posts,
                          prepare_personality,
                          pad_subreddits,
                          pad_triplet_baselines)


def triplet_transform(dataset, 
                      pad_to=[20,1,1],
                      batch_size=4, 
                      n_anchor=None,
                      filter_by_n=False):
    '''Transform pipeline for triplet dataset
    Args:
        dataset: dataset to transform
        pad_to (list): number of maximum posts to pad to
            for anchor, positive, negative example respectively 
        batch_size (int): global batch size (i.e., number of 
            replicas * effective batch size)
    '''
    dataset = pad_and_stack_triplet(dataset, 
                                    pad_to, 
                                    n_anchor,
                                    filter_by_n)
    return dataset.batch(batch_size, drop_remainder=True)


def classification_transform(dataset,
                             n_posts,
                             batch_size=4):
    '''Transform pipeline for classification dataset
    Args:
        dataset: dataset to transform
    '''
    dataset = stack_classification(dataset, n_posts)
    return dataset.batch(batch_size, drop_remainder=True)



def mlm_transform(dataset, 
                  is_context=True,
                  mask_proportion=.15,
                  batch_size=4,
                  is_combined=False):
    '''Transform pipeline for mlm dataset
    Args:
        dataset: dataset to transform
        is_context (bool): whether there is are context posts 
            or only the target is passed
        batch_size (int): global batch size (i.e., number of 
            replicas * effective batch size)
    '''
    dataset = mask_and_stack_mlm(dataset, is_context, 
                                 is_combined, mask_proportion)
    return dataset.batch(batch_size, drop_remainder=True)


def agg_transform(dataset,
                  targets,
                  batch_size=4):
    ''' Transform pipeline for aggregate prediction '''
    dataset = prepare_agg(dataset, targets)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


def personality_transform(dataset,
                          targets,
                          batch_size=4):
    ''' Transform pipeline for aggregate prediction '''
    dataset = prepare_personality(dataset, targets)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


def posts_transform(dataset,
                    targets,
                    batch_size=4):
    ''' Transform pipeline for aggregate prediction '''
    dataset = prepare_posts(dataset, targets)
    return dataset.batch(batch_size, drop_remainder=True)


def subreddit_transform(dataset, pad_to=3, 
                        batch_size=4, nr=3):  
    ''' Transform pipeline for subreddit prediction'''
    dataset = pad_subreddits(dataset,
                             pad_to, 
                             nr)
    return dataset.batch(batch_size, drop_remainder=True)    


def triplet_baselines_transform(dataset, pad_to=3, 
                                batch_size=4, nr=3,
                                dedict=False):  
    ''' Transform pipeline for subreddit prediction'''
    dataset = pad_triplet_baselines(dataset, pad_to, nr)
    if dedict:
        dataset = dataset.map(lambda x: (x['input_ids'], x['labels']))
    return dataset.batch(batch_size, drop_remainder=True)    

