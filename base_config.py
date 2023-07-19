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
                  drop_rate=0.0,
                  drop_path_rate=0.0,
                  include_top=True,
                  num_classes=1000,
                  block_layer="mixer_block",
                  mlp_layer="mlp",
                  act_layer = "gelu",
                  norm_layer="layer_norm"
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
    config.drop_path_rate = drop_path_rate
    config.drop_rate = drop_rate
    config.initializer_range = 0.1
    config.num_classes = num_classes
    config.name = config.model_name
    config.mlp_ratio = mlp_ratio

    config.n_channels = 3
    config.model_type = config.model_name
    config.include_top = include_top

    config.block_layer = block_layer
    config.mlp_layer = mlp_layer
    config.act_layer = act_layer
    config.norm_layer = norm_layer
    config.init_values = 1e-5

    return config.lock() 