from tensorflow import keras 
import tensorflow as tf 
from tensorflow.keras import * 
from tensorflow.keras.layers import * 
from .layers import norm_layer_factory, act_layer_factory, PatchEmbed
from .blocks import MixerBlock
from ml_collections import ConfigDict
from .block_layers import BLOCK_LAYERS
from typing import * 
import collections 
from collections import *

class MLPMixer(tf.keras.Model):

    def __init__(self, config: ConfigDict, *args, **kwargs):
        super(MLPMixer, self).__init__(*args, **kwargs)
        self.config = config
        self.projection_dim = config.projection_dim

        norm_layer = norm_layer_factory(config.norm_layer)
        block_layer = BLOCK_LAYERS[config.block_layer]

        self.stem = PatchEmbed(config, name="patch_embedding")

        self.blocks = [
            block_layer(config, name=f"blocks_{j}") for j in range(config.depth)
        ]

        self.norm = norm_layer(name="norm")

        self.head = (
            tf.keras.layers.Dense(units=config.num_classes, name="head")
            if config.num_classes > 0
            else tf.keras.layers.Activation("linear")
        )

    @property
    def dummy_inputs(self) -> tf.Tensor:
        return tf.zeros((1, *self.config.image_size, self.config.in_channels))

    @property
    def feature_names(self) -> List[str]:
        return (
            ["stem"]
            + [f"block_{j}" for j in range(self.config.depth)]
            + ["features_all", "features", "logits"]
        )

    def forward_features(self, x, training=False, return_features=False):
        features = {}
        x = self.stem(x, training=training)
        features["stem"] = x

        for j, block in enumerate(self.blocks):
            x = block(x, training=training)
            features[f"block_{j}"] = x

        x = self.norm(x, training=training)
        features["features_all"] = x

        x = tf.reduce_mean(x, axis=1)
        features["features"] = x

        return (x, features) if return_features else x

    def call(self, x, training=False, return_features=False):
        features = {}
        x = self.forward_features(x, training, return_features)
        if return_features:
            x, features = x
        x = self.head(x)
        features["logits"] = x
        return (x, features) if return_features else x