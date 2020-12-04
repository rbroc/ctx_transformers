import tensorflow as tf
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def create_ckpt_callback(ds_name, ds_dir='datasets', 
                         ckpt_name='cp.ckpt', **kwargs):
    ''' Create checkpoint callback for a Keras model'''
    outdir = Path(ds_dir) / ds_name / 'checkpoints' /  ckpt_name
    outdir.mkdir(parents=True, exist_ok=True)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=str(outdir),
                                                     **kwargs)
    return cp_callback
    

def save_predictions(ds_name, tokenizer, test_ds, pred_lab,
                     lmap, nr_tkns, nr_ex=100, ds_dir='datasets', 
                     out=False):
    ''' Save sample predictions in tsv files'''
    outdir = Path(ds_dir) / ds_name / 'examples'
    outdir.mkdir(parents=True, exist_ok=True)
    corr_fname = outdir / f'correct_{nr_ex}.txt'
    incorr_fname = outdir / f'incorrect_{nr_ex}.txt'
    ctxt, clab, inctxt, tlab, plab = [], [], [], [], []
    for idx, ex in enumerate(test_ds.take(nr_ex)):
        pred = pred_lab[idx].numpy()
        true = ex[1].numpy()
        post = tokenizer.decode(ex[0]['input_ids'].numpy()).split()
        if pred != true:
            inctxt.append(' '.join(post[1:nr_tkns]))
            plab.append(lmap[str(pred_lab[idx].numpy())])
            tlab.append(lmap[str(ex[1].numpy())])
        else:
            ctxt.append(' '.join(post[1:nr_tkns]))
            clab.append(lmap[str(ex[1].numpy())])
    inc_df = pd.DataFrame(zip(inctxt, tlab, plab), 
                          columns=['text', 'pred', 'true'])
    inc_df.to_csv(incorr_fname, sep='\t', index=False)
    c_df = pd.DataFrame(zip(ctxt, clab),
                        columns=['text', 'true'])
    c_df.to_csv(corr_fname, sep='\t', index=False)
    if out:
        return zip(ctxt, clab), zip(inctxt, plab, tlab)

def save_confusion_matrix(conf_mat, labs, 
                          ds_name, mname, 
                          ds_dir='datasets',
                          save=True, 
                          map_labels=False,
                          lab_dict=None,
                          figsize=(20,20),
                          **kwargs):
    '''Save confusion matrix'''
    outdir = Path(ds_dir) / ds_name / 'figures'
    outdir.mkdir(parents=True, exist_ok=True)
    fname = outdir / str(f'confmat_{mname}.png')
    if map_labels:
        labs = [lab_dict[str(l)] for l in labs]
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(conf_mat, 
                annot=True, fmt='d',
                cmap='viridis',
                xticklabels=labs, 
                yticklabels=labs)
    plt.ylabel('true')
    plt.xlabel('predicted')
    if save:
        plt.savefig(fname)
    else:
        plt.show()

