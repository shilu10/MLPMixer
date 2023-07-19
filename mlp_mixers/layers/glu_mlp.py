import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import * 
from tensorflow.keras.layers import * 
from .factory import act_layer_factory


class GatedBiasInitializer(tf.keras.initializers.Initializer):
    """
    Splits tensor in half along last axis (channels). Initializes second half with
    ones.

    Used for bias term in Gated Linear Units.
    """

    def __init__(self, initializer="zeros"):
        self.initializer = tf.keras.initializers.get(initializer)

    def __call__(self, shape, dtype=None, **kwargs):
        if dtype is None:
            dtype = tf.keras.backend.floatx()

        assert shape[-1] % 2 == 0
        split_shape = shape[:-1] + [
            shape[-1] // 2,
        ]
        x1 = self.initializer(split_shape, dtype=dtype)
        x2 = tf.ones(split_shape, dtype=dtype)
        x = tf.concat([x1, x2], axis=-1)
        return x


class GatedKernelInitializer(tf.keras.initializers.Initializer):
    """
    Splits tensor in half along last axis (channels). Initializes second half with
    normal distribution with stddev=1e-6.

    Used for kernel term in Gated Linear Units.
    """

    def __init__(self, initializer="glorot_uniform"):
        self.normal = tf.keras.initializers.RandomNormal(stddev=1e-6)
        self.initializer = tf.keras.initializers.get(initializer)

    def __call__(self, shape, dtype=None, **kwargs):
        if dtype is None:
            dtype = tf.keras.backend.floatx()

        assert shape[-1] % 2 == 0
        split_shape = shape[:-1] + [
            shape[-1] // 2,
        ]
        x1 = self.initializer(split_shape, dtype=dtype)
        x2 = self.normal(split_shape, dtype=dtype)
        x = tf.concat([x1, x2], axis=-1)
        return x


class GluMLP(tf.keras.layers.Layer):
    """
    MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202
    """
    def __init__(
        self,
        hidden_dim: int,
        projection_dim: int,
        drop_rate: float,
        act_layer: str,
        **kwargs,
    ):
        super(GluMLP, self).__init__(**kwargs)
        act_layer = act_layer_factory(act_layer)
        assert hidden_dim % 2 == 0

        self.fc1 = tf.keras.layers.Dense(
            units=hidden_dim,
            bias_initializer=GatedBiasInitializer(),
            kernel_initializer=GatedKernelInitializer(),
            name="fc1",
        )
        self.act = act_layer()
        self.drop1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.fc2 = tf.keras.layers.Dense(units=projection_dim, name="fc2")
        self.drop2 = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training=False):
        x = self.fc1(x)
        x, gates = tf.split(x, num_or_size_splits=2, axis=-1)
        x = x * self.act(gates)
        x = self.drop1(x, training=training)
        x = self.fc2(x)
        x = self.drop2(x, training=training)
        return x