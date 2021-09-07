LOG_DICT = {'triplet': ['losses', 
                        'metrics', 
                        'dist_pos', 
                        'dist_neg', 
                        'dist_anch'],
           'mlm': ['losses', 'entropy', 'is_true_top'],
           'mlm_simple': ['losses', 'entropy', 'is_true_top'],
           'agg': ['losses']}


META_DICT = {'triplet': [],
             'mlm': [], # labels
             'mlm_simple': [],
             'agg': []} # labels


PBAR_DICT = {'triplet': ['losses', 'metrics'],
             'mlm': ['losses', 'entropy', 'is_true_top'],
             'mlm_simple': ['losses', 'entropy', 'is_true_top'],
             'agg': ['losses']}