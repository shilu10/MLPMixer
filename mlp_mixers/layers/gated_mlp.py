import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import * 
from tensorflow.keras.layers import * 


class SpatialGatingUnit(tf.keras.layers.Layer):
    """
    Spatial Gating Unit

    Based on: Pay Attention to MLPs - https://arxiv.org/abs/2105.08050
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        norm_layer = norm_layer_factory("layer_norm")
        self.norm = norm_layer(name="norm")

    def build(self, input_shape):
        seq_len = input_shape[-2]
        self.proj = tf.keras.layers.Dense(
            units=seq_len,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-6),
            bias_initializer=tf.keras.initializers.Ones(),
            name="proj",
        )

    def call(self, x, training=False):
        u, v = tf.split(x, num_or_size_splits=2, axis=-1)

        v = self.norm(v, training=training)
        v = tf.transpose(v, perm=(0, 2, 1))
        v = self.proj(v)
        v = tf.transpose(v, perm=(0, 2, 1))
        x = u * v
        return x


class GatedMLP(tf.keras.layers.Layer):
    """MLP as used in gMLP"""

    def __init__(
        self,
        hidden_dim: int,
        projection_dim: int,
        drop_rate: float,
        act_layer: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        act_layer = act_layer_factory(act_layer)

        self.fc1 = tf.keras.layers.Dense(units=hidden_dim, name="fc1")
        self.act = act_layer()
        self.drop1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.gate = SpatialGatingUnit(name="gate")
        self.fc2 = tf.keras.layers.Dense(units=projection_dim, name="fc2")
        self.drop2 = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x, training=training)
        x = self.gate(x, training=training)
        x = self.fc2(x)
        x = self.drop2(x, training=training)
        return x