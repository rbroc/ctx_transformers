LOG_DICT = {'triplet': ['losses', 
                        'metrics', 
                        'dist_pos', 
                        'dist_neg', 
                        'dist_anch'],
           'personality': ['losses'],
           'classification': ['losses', 'metrics', 'probs'],
           'subreddit_classification': ['losses', 'recall', 'precision', 'accuracy', 
                                        'tp', 'tn', 'fp', 'fn'],
           'mlm': ['losses', 'entropy', 'is_true_top'],
           'mlm_combined': ['losses', 'entropy', 'is_true_top'],
           'mlm_simple': ['losses', 'entropy', 'is_true_top']}

META_DICT = {'triplet': [],
             'classification': ['labels'],
             'subreddit_classification': [],
             'mlm': [],
             'mlm_combined': [],
             'mlm_simple': [],
             'personality': ['labels'],
             'agg': ['labels'],
             'posts':['labels']}

PBAR_DICT = {'triplet': ['losses', 'metrics'],
             'classification': ['losses', 'metrics', 'probs'],
             'subreddit_classification': ['losses', 'recall', 'precision', 'accuracy', 
                                          'tp', 'tn', 'fp', 'fn'],
             'mlm': ['losses', 'entropy', 'is_true_top'],
             'mlm_combined': ['losses', 'entropy', 'is_true_top'],
             'mlm_simple': ['losses', 'entropy', 'is_true_top'],
             'personality': ['losses'],
             'agg': ['losses'],
             'posts': ['losses']}
# losses accuracy