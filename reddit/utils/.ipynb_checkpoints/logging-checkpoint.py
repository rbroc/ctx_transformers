LOG_DICT = {'triplet': ['losses', 
                        'metrics', 
                        'dist_pos', 
                        'dist_neg', 
                        'dist_anch'],
           'classification': ['losses', 'metrics', 'probs'],
           'mlm': ['losses', 'entropy', 'is_true_top'],
           'mlm_simple': ['losses', 'entropy', 'is_true_top']}


META_DICT = {'triplet': [],
             'classification': ['labels'],
             'mlm': [],
             'mlm_simple': [],
             'agg': ['labels'],
             'posts':['labels']}


PBAR_DICT = {'triplet': ['losses', 'metrics'],
             'classification': ['losses', 'metrics', 'probs'],
             'mlm': ['losses', 'entropy', 'is_true_top'],
             'mlm_simple': ['losses', 'entropy', 'is_true_top'],
             'agg': ['losses'],
             'posts': ['losses']}