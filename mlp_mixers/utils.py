import tensorflow as tf
from tensorflow import keras
import collections
from ml_collections import ConfigDict
from .layers import MLP
from .blocks import MixerBlock

BLOCK_LAYERS = {
    "mixer_block": MixerBlock}

MLP_LAYERS = {
    "mlp": MLP
}