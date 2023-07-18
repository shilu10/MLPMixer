from utils import get_initializer
from tensorflow import keras 
import tensorflow as tf 
import numpy as np 
from tensorflow.keras import layers 
from tensorflow.keras.layers import * 
from ml_collections import ConfigDict 
from timm.layers import to_2tuple
from typing import *


class PatchEmbed(keras.layers.Layer):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """
    def __init__(self, config: ConfigDict, **kwargs):
        super(PatchEmbed, self).__init__(**kwargs)
        image_size = config.image_size
        patch_size = config.patch_size
        projection_dim = config.projection_dim
        n_channels = config.n_channels

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = ((image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1]))

        # calculation of num of patches
        self.num_patches = num_patches
        self.config = config
        self.image_size = image_size
        self.n_channels = n_channels
        self.projection_dim = projection_dim
        self.patch_size = patch_size

        # patch generator
        self.projection = Conv2D(
            kernel_size=patch_size,
            strides=patch_size,
            data_format="channels_last",
            filters=projection_dim,
            padding="valid",
            use_bias=True,
            kernel_initializer=get_initializer(self.config.initializer_range),
            bias_initializer="zeros",
            name="projection"
        )

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        shape = tf.shape(x)
        batch_size, height, width, n_channel = shape[0], shape[1], shape[2], shape[3]

        projection = self.projection(x)
        embeddings = tf.reshape(tensor=projection, shape=(batch_size, self.num_patches, -1))

        return embeddings


# Referred from: github.com:rwightman/pytorch-image-models.
class StochasticDepth(layers.Layer):
    def __init__(self, drop_prop, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prop

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x


class FNN(keras.layers.Layer):
  def __init__(self, dropout_rate, mlp_units):
    super(FNN, self).__init__()
    self.dropout_rate = dropout_rate
    self.mlp_units = mlp_units

    self.fc1 = Dense(mlp_units[0])
    self.fc2 = Dense(mlp_units[1])
    self.dropout = Dropout(rate=self.dropout_rate)

  def call(self, inputs, training=False):
    x = self.fc1(inputs)
    x = tf.nn.gelu(x)
    x = self.dropout(x)
    x = self.fc2(x)
    x = self.dropout(x)

    return x


class MixerBlock(keras.Model):
    def __init__(self, config: ConfigDict, droppath_rate, **kwargs):

        super(MixerBlock, self).__init__(**kwargs)

        tokens_dim, channels_dim = [int(x * config.projection_dim) for x in to_2tuple(config.mlp_ratio)]
        token_units = [tokens_dim, config.num_patches]
        channel_units = [channels_dim, config.projection_dim]

        print(token_units, channel_units)

        self.mlp_tokens = FNN(config.dropout_rate, token_units)
        self.mlp_channels = FNN(config.dropout_rate, channel_units)
        self.norm1 = LayerNormalization(epsilon=config.layernorm_eps)
        self.norm2 = LayerNormalization(epsilon=config.layernorm_eps)
        self.drop_path = StochasticDepth(droppath_rate) if droppath_rate > 0. else tf.identity

    def call(self, x):
        x1 = self.norm1(x)
        x1 = tf.transpose(x1, [0, 2, 1])
        x2 = self.mlp_tokens(x1)
        x3 = tf.transpose(x2, [0, 2, 1]) + x
        # dropath
        x3 = self.drop_path(x3)

        x4 = self.norm2(x3)
        outputs = self.mlp_channels(x4) + x3
        # droppath
        outputs = self.drop_path(outputs)

        return outputs


class MLPMixer(keras.Model):

    def __init__(self, config: ConfigDict, **kwargs):

        super(MLPMixer, self).__init__( **kwargs)

        assert config.image_size % config.patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.include_top = config.include_top
        self.num_patches =  (config.image_size// config.patch_size) ** 2
        self.stem = PatchEmbed(config, name="patch_embed")

        # mlp mixer blocks
        self.mixer_blocks = []
        for _ in range(config.depth):
            self.mixer_blocks.append(MixerBlock(config, config.droppath_rate, name=f"mixer_block{_}"))


        # other layers
        self.dropout = Dropout(rate=config.dropout_rate)
        self.layer_norm = LayerNormalization(epsilon=config.layernorm_eps)
        self.gap = GlobalAveragePooling1D()

        if config.include_top:
          self.head = Dense(
                  config.num_classes,
                  kernel_initializer="zeros",
                  dtype="float32",
                  name="classification_head",
            )
        self.config = config

    def call(self, x, training=False):

        x = self.stem(x)

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)
        x = self.dropout(x, training=training)
        
        if self.config.global_pool == "avg":
          #x = tf.reduce_mean(x, axis=1)
          x = self.gap(x)
          
        if self.include_top:
          return self.head(x)

        return x