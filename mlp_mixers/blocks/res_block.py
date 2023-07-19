import sys, os, shutil

import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import * 
from tensorflow.keras.layers import * 
from ..layers import norm_layer_factory, act_layer_factory, DropPath
from ..mlp_layers import MLP_LAYERS
from ml_collections import ConfigDict


class ResBlock(tf.keras.layers.Layer):
    """
    Residual MLP block with LayerScale

    Based on: ResMLP: Feedforward networks for image classification...
    """

    def __init__(self, config: ConfigDict, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.config = config

        norm_layer = norm_layer_factory(config.norm_layer)
        mlp_layer = MLP_LAYERS[config.mlp_layer]
        self.norm1 = norm_layer(name="norm1")
        self.linear_tokens = tf.keras.layers.Dense(
            units=config.num_patches,
            name="linear_tokens",
        )
        self.drop_path = DropPath(drop_prob=config.drop_path_rate)
        self.norm2 = norm_layer(name="norm2")
        self.mlp_channels = mlp_layer(
            hidden_dim=int(config.projection_dim * config.mlp_ratio[1]),
            projection_dim=config.projection_dim,
            drop_rate=config.drop_rate,
            act_layer=config.act_layer,
            name="mlp_channels",
        )

    def build(self, input_shape):
        self.ls1 = self.add_weight(
            shape=(self.config.projection_dim,),
            initializer=tf.keras.initializers.Constant(self.config.init_values),
            trainable=True,
            name="ls1",
        )
        self.ls2 = self.add_weight(
            shape=(self.config.projection_dim,),
            initializer=tf.keras.initializers.Constant(self.config.init_values),
            trainable=True,
            name="ls2",
        )

    def call(self, x, training=False):
        shortcut = x
        x = self.norm1(x, training=training)
        x = tf.transpose(x, perm=(0, 2, 1))
        x = self.linear_tokens(x, training=training)
        x = tf.transpose(x, perm=(0, 2, 1))
        x = self.ls1 * x
        x = self.drop_path(x, training=training)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x, training=training)
        x = self.mlp_channels(x, training=training)
        x = self.ls2 * x
        x = self.drop_path(x, training=training)
        x = x + shortcut
        return x