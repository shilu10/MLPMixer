import tensorflow as tf 
import tensorflow.keras as keras 
from tensorflow.keras.layers import * 
from tensorflow.keras import * 
from .factory import act_layer_factory, norm_layer_factory


class MLP(tf.keras.layers.Layer):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        hidden_dim: int,
        projection_dim: int,
        drop_rate: float,
        act_layer: str,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        **kwargs,
    ):
        super(MLP, self).__init__(**kwargs)
        act_layer = act_layer_factory(act_layer)

        self.fc1 = tf.keras.layers.Dense(
            units=hidden_dim,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name="fc1",
        )
        self.act = act_layer()
        self.drop1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.fc2 = tf.keras.layers.Dense(
            units=projection_dim,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name="fc2",
        )
        self.drop2 = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x, training=training)
        x = self.fc2(x)
        x = self.drop2(x, training=training)
        return x