import sys, os, shutil
sys.path.append("MLPMixer1/mlp_mixers/layers/")

import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import * 
from tensorflow.keras.layers import * 
from layers import norm_layer_factory, act_layer_factory, DropPath
from utils import MLP_LAYERS
from ml_collections import ConfigDict


class MixerBlock(tf.keras.layers.Layer):
    """
    Residual Block w/ token mixing and channel MLPs
    Based on: "MLP-Mixer: An all-MLP Architecture for Vision"
    """

    def __init__(self, config: ConfigDict, **kwargs):
        super(MixerBlock, self).__init__(**kwargs)
        self.config = config

        norm_layer = norm_layer_factory(config.norm_layer)
        mlp_layer = MLP_LAYERS[config.mlp_layer]
        tokens_dim, channels_dim = [int(x * config.projection_dim) for x in config.mlp_ratio]

        self.norm1 = norm_layer(name="norm1")
        self.mlp_tokens = mlp_layer(
            hidden_dim=tokens_dim,
            projection_dim=config.num_patches,
            drop_rate=config.drop_rate,
            act_layer=config.act_layer,
            name="mlp_tokens",
        )
        self.drop_path = DropPath(drop_prob=config.drop_path_rate)
        self.norm2 = norm_layer(name="norm2")
        self.mlp_channels = mlp_layer(
            hidden_dim=channels_dim,
            projection_dim=config.projection_dim,
            drop_rate=config.drop_rate,
            act_layer=config.act_layer,
            name="mlp_channels",
        )

    def call(self, x, training=False):
        shortcut = x
        x = self.norm1(x, training=training)
        x = tf.transpose(x, perm=(0, 2, 1))
        x = self.mlp_tokens(x, training=training)
        x = tf.transpose(x, perm=(0, 2, 1))
        x = self.drop_path(x, training=training)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x, training=training)
        x = self.mlp_channels(x, training=training)
        x = self.drop_path(x, training=training)
        x = x + shortcut

        return x