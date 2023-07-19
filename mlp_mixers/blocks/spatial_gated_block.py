import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import * 
from tensorflow.keras.layers import * 
from ..layers import GatedMLP


class SpatialGatingBlock(tf.keras.layers.Layer):
    """
    Residual Block with Spatial Gating

    Based on: Pay Attention to MLPs - https://arxiv.org/abs/2105.08050
    """

    def __init__(self, config: ConfigDict, **kwargs):
        super(SpatialGatingBlock, self).__init__(**kwargs)
        self.config = config

        norm_layer = norm_layer_factory(config.norm_layer)
        mlp_layer = MLP_LAYERS[config.mlp_layer]
        self.norm = norm_layer(name="norm")
        self.mlp_channels = mlp_layer(
            hidden_dim=int(config.projection_dim * config.mlp_ratio[1]),
            projection_dim=config.projection_dim,
            drop_rate=config.drop_rate,
            act_layer=config.act_layer,
            name="mlp_channels",
        )
        self.drop_path = DropPath(drop_prob=config.drop_path_rate)

    def call(self, x, training=False):
        shortcut = x
        x = self.norm(x, training=training)
        x = self.mlp_channels(x, training=training)
        x = self.drop_path(x, training=training)
        x = x + shortcut
        return x