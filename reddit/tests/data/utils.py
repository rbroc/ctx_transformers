import tensorflow as tf

def duplicate_examples(dataset):
    dataset = dataset.map(lambda x: {'iids': x['iids'],
                                     'amask': x['amask'],
                                     'pos_iids': tf.concat([x['pos_iids'],
                                                            x['neg_iids']], 
                                                            axis=0),
                                     'pos_amask': tf.concat([x['pos_amask'],
                                                             x['neg_amask']], 
                                                             axis=0),
                                     'neg_iids': tf.concat([x['neg_iids'],
                                                            x['pos_iids']], 
                                                            axis=0),
                                     'neg_amask': tf.concat([x['neg_amask'],
                                                             x['pos_amask']], 
                                                             axis=0),
                                     'author_id': x['author_id']})
    return dataset