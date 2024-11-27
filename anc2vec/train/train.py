#!/usr/bin/env python3
import sys
import datetime
import tempfile

import tensorflow as tf
import wandb

from . import onto
from .utils import Tokenizer
from .models import Embedder
from .dataset import Dataset

def define_callbacks(model_name, use_wandb=True):
    tmpdir = tempfile.gettempdir()
    model_file = tmpdir + '/models/' + model_name + '/' + 'best.keras'

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_file, save_best_only=True, monitor='loss')
    ]

    if use_wandb:
        # Create a custom callback for wandb logging
        class CustomWandbCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                wandb.log(logs or {})

        callbacks.append(CustomWandbCallback())

    return callbacks

def fit(obo_fin, embeddings_path, embedding_sz=200, batch_sz=64, num_epochs=100,
        loss_weights=None, use_lr_schedule=False, initial_lr=0.001):
    go = onto.Ontology(obo_fin, with_rels=True, include_alt_ids=False)
    tok = Tokenizer(go)

    buffer_sz = tok.vocab_sz
    dataset = Dataset(tok,
                     embeddings_path,
                     batch_sz,
                     buffer_sz,
                     shuffle=True,
                     seed=1234)
    train_set = dataset.build()
    train_set = train_set.take(tok.vocab_sz).cache()

    model = Embedder.build(
        tok.vocab_sz,
        embedding_sz,
        loss_weights=loss_weights,
        use_lr_schedule=use_lr_schedule,
        initial_lr=initial_lr
    )
    print(model.summary())

    model_name = (f"{model.name}_embedding_sz={embedding_sz}"
                 f"_lr={initial_lr}_schedule={use_lr_schedule}")

    model.fit(
        train_set,
        epochs=num_epochs,
        callbacks=define_callbacks(model_name)
    )

    # recover trained model with best loss
    tmpdir = tempfile.gettempdir()
    model_file = tmpdir + '/models/' + model_name + '/best.keras'

    model = tf.keras.models.load_model(model_file, compile=False)
    embeddings = model.get_layer('embedding').weights[0].numpy()

    # transform embeddings into a dictionary
    embds = {}
    for k, v in dataset.embeddings.items():
        embds[k] = v @ embeddings

    return embds
