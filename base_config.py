from ml_collections import ConfigDict 
from tensorflow import keras 
import tensorflow as tf 
import numpy as np 


def get_baseconfig(model_type="mixer_l16_224.goog_in21k_ft_in1k", 
                  image_size=224, 
                  patch_size=16, 
                  depth=24,
                  projection_dim=1024,
                  mlp_ratio=(0.5, 4.0),
                  init_values=None,
                  dropout_rate=0.0,
                  droppath_rate=0.0,
                  include_top=True,
                  num_classes=1000,
                  layernorm_eps=1e-5
            ):

    config = ConfigDict()

    # base config (common for all model type)
    config.model_name = model_type
    config.patch_size = patch_size
    config.image_size = image_size
    config.num_patches = pow(config.image_size // config.patch_size, 2)
    config.depth = depth
    config.projection_dim = projection_dim
    config.input_shape = (config.image_size, config.image_size, 3)
    config.init_values = init_values
    config.droppath_rate = droppath_rate
    config.dropout_rate = dropout_rate
    config.initializer_range = 0.02
    config.layernorm_eps = 1e-5
    config.num_classes = num_classes
    config.name = config.model_name
    config.global_pool = "avg"
    config.mlp_ratio = mlp_ratio

    config.n_channels = 3
    config.model_type = config.model_name
    config.include_top = True

    return config.lock() 