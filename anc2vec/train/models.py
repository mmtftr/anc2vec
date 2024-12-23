import tensorflow as tf
import numpy as np

from . import losses
from .layers import Distance2logprob

class Embedder(tf.keras.Model):
    def _build(vocab_sz, embedding_sz):
        inputs = tf.keras.Input(shape=(768,), dtype=tf.float32)

        hidden = tf.keras.layers.Dense(
            units=embedding_sz,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1234),
            use_bias=False,
            name='embedding')(inputs)

        y = tf.keras.layers.Dense(
            vocab_sz,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1234))(hidden)
        y = tf.keras.layers.Activation('softmax', name='ance')(y)

        # predict namespace (BP, CC, & MF)
        z = tf.keras.layers.Dense(
            3,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1234))(hidden)
        z = tf.keras.layers.Activation('softmax', name='name')(z)

        w = tf.keras.layers.Dense(
            vocab_sz,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1234))(hidden)
        w = tf.keras.layers.Activation('softmax', name='auto')(w)

        return tf.keras.Model(inputs, [y, z, w])

    def build(vocab_sz, embedding_sz, loss_weights=None, use_lr_schedule=False, initial_lr=0.001):
        model = Embedder._build(vocab_sz, embedding_sz)

        # Default loss weights if none provided
        if loss_weights is None:
            loss_weights = {
                'ance': 1.0,
                'name': 1.0,
                'auto': 1.0
            }

        # Configure learning rate schedule if enabled
        if use_lr_schedule:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_lr,
                decay_steps=1000,
                decay_rate=0.9
            )
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)

        model.compile(
            optimizer=optimizer,
            loss=[
                losses.Word2vecLoss(),
                tf.keras.losses.CategoricalCrossentropy(),
                losses.Word2vecLoss(),
            ],
            loss_weights=[
                loss_weights['ance'],
                loss_weights['name'],
                loss_weights['auto']
            ],
            metrics={
                'ance': tf.keras.metrics.Recall(name='rc'),
                'name': tf.keras.metrics.CategoricalAccuracy(name='ac'),
                'auto': tf.keras.metrics.MeanSquaredError(name='ms'),
            }
        )

        return model

    @tf.function
    def test_step(self, batch):
        x, y = batch
        y_pred = self(x, training=False)
        loss = self.compiled_loss(y, y_pred)

        # update loss
        self.compiled_metrics.update_state(y, y_pred, [])
        self.metrics[4].update_state(y[0], y_pred[0]) # neighbors
        self.metrics[5].update_state(y[1], y_pred[1]) # namespace
        self.metrics[6].update_state(y[2], y_pred[2]) # auto

        return { m.name: m.result() for m in self.metrics }

    @tf.function
    def train_step(self, batch):
        x, y = batch

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True) # forward pass
            loss = self.compiled_loss(y, y_pred)
            #loss +=  self.foo_loss(y_pred[0], y[1]) # second-order NCA

        # compute gradients
        variables = self.trainable_variables
        grad = tape.gradient(loss, variables)
        # update weights
        self.optimizer.apply_gradients(zip(grad, variables))

        # update loss
        self.compiled_metrics.update_state(y, y_pred, [])
        self.metrics[4].update_state(y[0], y_pred[0]) # neighbors
        self.metrics[5].update_state(y[1], y_pred[1]) # namespace
        self.metrics[6].update_state(y[2], y_pred[2]) # auto

        return { m.name: m.result() for m in self.metrics }

    def reset_states(self):
        super().reset_metrics()
        for m in self.metrics:
            m.reset_states()
        for m in self.my_metrics:
            m.reset_states()
