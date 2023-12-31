import numpy as np 
import tensorflow as tf 
from tensorflow.keras import * 
from tensorflow import keras 
from tensorflow.keras.layers import * 
import os, sys, shutil, glob 
import timm 
from .utils import modify_tf_block, get_tf_qkv
from .mlp_mixers.main import MLPMixer
from .base_config import get_baseconfig
import yaml


def port(model_type: str = "mixer_b16_224.goog_in21k_ft_in1k", 
         model_savepath: str = 'models/', 
         include_top: bool = True
    ):

    print("Instantiating PyTorch model...")
    pt_model = timm.create_model(
        model_name=model_type, 
        num_classes=1000, 
        pretrained=True
    )
    
    pt_model.eval()

    print("Instantiating TF model...")
    model_cls = MLPMixer

    # replace(".", "_")
    config_file_path = f'configs/{model_type}.yaml'
    with open(config_file_path, "r") as f:
        data = yaml.safe_load(f)

    config = get_baseconfig(
        model_type = model_type,
        image_size = data.get("image_size"),
        patch_size = data.get("patch_size"),
        depth = data.get("depth"),
        projection_dim = data.get("projection_dim"),
        include_top = include_top,
        mlp_ratio = data.get("mlp_ratio"),
        mlp_layer = data.get("mlp_layer"),
        block_layer = data.get("block_layer"),
        act_layer = data.get('act_layer'),
        norm_layer = data.get('norm_layer')
    )

    tf_model = model_cls(config)

    #print(tf_model.positional_embedding, tf_model.cls_token)
    image_dim = data.get("image_size")
    dummy_inputs = tf.ones((2, image_dim, image_dim, 3))
    _ = tf_model(dummy_inputs)[0]

    if include_top:
        assert tf_model.count_params() == sum(
            p.numel() for p in pt_model.parameters()
        )

    # Load the PT params.
    pt_model_dict = pt_model.state_dict()
    pt_model_dict = {k: pt_model_dict[k].numpy() for k in pt_model_dict}

    print("Beginning parameter porting process...")

    # head Norm layer.
    if "res" in model_type:
        tf_model.layers[-2].alpha = tf.Variable(np.ravel(pt_model_dict["norm.alpha"]))
        tf_model.layers[-2].beta = tf.Variable(np.ravel(pt_model_dict[f"norm.beta"]))

    else:
        tf_model.layers[-2] = modify_tf_block(
            tf_model.layers[-2],
            pt_model_dict["norm.weight"],
            pt_model_dict["norm.bias"]
        )

    # patch embedding.
    tf_model.layers[0].projection = modify_tf_block(
        tf_model.layers[0].projection,
        pt_model_dict["stem.proj.weight"],
        pt_model_dict["stem.proj.bias"]
    )

    # Head layers.
    if include_top:
        tf_model.layers[-1] = modify_tf_block(
            tf_model.layers[-1],
            pt_model_dict["head.weight"],
            pt_model_dict["head.bias"]
        )

    # mixer layers:
    if "mixer" in model_type:
        tf_model = convert_mlpmixer_layer(tf_model=tf_model, 
                                          pt_model_dict=pt_model_dict, 
                                          config=config
                                        )

    elif "gmlp" in model_type:
        tf_model = convert_gatedmlp_layer(tf_model=tf_model, 
                                          pt_model_dict=pt_model_dict, 
                                          config=config
                                        )

    elif "resmlp" in model_type:
        tf_model = convert_resmlp_layer(tf_model=tf_model, 
                                          pt_model_dict=pt_model_dict, 
                                          config=config
                                        )

    print("Porting successful, serializing TensorFlow model...")

    save_path = os.path.join(model_savepath, model_type)
    save_path = f"{save_path}_fe" if not include_top else save_path
    tf_model.save(save_path)
    print(f"TensorFlow model serialized at: {save_path}...")


def convert_mlpmixer_layer(tf_model, pt_model_dict, config):

    for indx, layer in enumerate(tf_model.layers[1: config.depth+1]):
        pt_block_name = f"blocks.{indx}"

        # mlp_channels
        layer.mlp_channels.fc1 = modify_tf_block(
                layer.mlp_channels.fc1,
                pt_model_dict[f"{pt_block_name}.mlp_channels.fc1.weight"],
                pt_model_dict[f"{pt_block_name}.mlp_channels.fc1.bias"]
            )

        layer.mlp_channels.fc2 = modify_tf_block(
                layer.mlp_channels.fc2,
                pt_model_dict[f"{pt_block_name}.mlp_channels.fc2.weight"],
                pt_model_dict[f"{pt_block_name}.mlp_channels.fc2.bias"]
            )

        # mlp_tokens
        layer.mlp_tokens.fc1 = modify_tf_block(
                layer.mlp_tokens.fc1,
                pt_model_dict[f"{pt_block_name}.mlp_tokens.fc1.weight"],
                pt_model_dict[f"{pt_block_name}.mlp_tokens.fc1.bias"]
            )

        layer.mlp_tokens.fc2 = modify_tf_block(
                layer.mlp_tokens.fc2,
                pt_model_dict[f"{pt_block_name}.mlp_tokens.fc2.weight"],
                pt_model_dict[f"{pt_block_name}.mlp_tokens.fc2.bias"]
            )

        # normalization
        layer.norm1 = modify_tf_block(
                layer.norm1,
                pt_model_dict[f"{pt_block_name}.norm1.weight"],
                pt_model_dict[f"{pt_block_name}.norm1.bias"]
            )

        layer.norm2 = modify_tf_block(
                layer.norm2,
                pt_model_dict[f"{pt_block_name}.norm2.weight"],
                pt_model_dict[f"{pt_block_name}.norm2.bias"]
            ) 
    
    return tf_model


def convert_gatedmlp_layer(tf_model, pt_model_dict, config):

    for indx, layer in enumerate(tf_model.layers[1: config.depth+1]):
        pt_block_name = f"blocks.{indx}"

        # mlp_channels
        layer.mlp_channels.fc1 = modify_tf_block(
                layer.mlp_channels.fc1,
                pt_model_dict[f"{pt_block_name}.mlp_channels.fc1.weight"],
                pt_model_dict[f"{pt_block_name}.mlp_channels.fc1.bias"]
            )

        layer.mlp_channels.fc2 = modify_tf_block(
                layer.mlp_channels.fc2,
                pt_model_dict[f"{pt_block_name}.mlp_channels.fc2.weight"],
                pt_model_dict[f"{pt_block_name}.mlp_channels.fc2.bias"]
            )

        # mlp_channels -. gate
        layer.mlp_channels.gate.proj = modify_tf_block(
                layer.mlp_channels.gate.proj,
                pt_model_dict[f"{pt_block_name}.mlp_channels.gate.proj.weight"],
                pt_model_dict[f"{pt_block_name}.mlp_channels.gate.proj.bias"]
            )
        
        # gate norm
        layer.mlp_channels.gate.norm = modify_tf_block(
                layer.mlp_channels.gate.norm,
                pt_model_dict[f"{pt_block_name}.mlp_channels.gate.norm.weight"],
                pt_model_dict[f"{pt_block_name}.mlp_channels.gate.norm.bias"]
            )

        # normalization (main)
        layer.norm = modify_tf_block(
                layer.norm,
                pt_model_dict[f"{pt_block_name}.norm.weight"],
                pt_model_dict[f"{pt_block_name}.norm.bias"]
            ) 
    
    return tf_model


def convert_resmlp_layer(tf_model, pt_model_dict, config):
    for indx, layer in enumerate(tf_model.layers[1: config.depth+1]):
        pt_block_name = f"blocks.{indx}"

        # mlp_channels
        layer.mlp_channels.fc1 = modify_tf_block(
                layer.mlp_channels.fc1,
                pt_model_dict[f"{pt_block_name}.mlp_channels.fc1.weight"],
                pt_model_dict[f"{pt_block_name}.mlp_channels.fc1.bias"]
            )

        layer.mlp_channels.fc2 = modify_tf_block(
                layer.mlp_channels.fc2,
                pt_model_dict[f"{pt_block_name}.mlp_channels.fc2.weight"],
                pt_model_dict[f"{pt_block_name}.mlp_channels.fc2.bias"]
            )

        # linear_tokens
        layer.linear_tokens.kernel.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.linear_tokens.weight"].transpose()))
        layer.linear_tokens.bias.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.linear_tokens.bias"]))

        # affine norm
        layer.norm1.alpha = tf.Variable(np.ravel(pt_model_dict[f"{pt_block_name}.norm1.alpha"]))
        layer.norm1.beta = tf.Variable(np.ravel(pt_model_dict[f"{pt_block_name}.norm1.beta"]))

        layer.norm2.alpha = tf.Variable(np.ravel(pt_model_dict[f"{pt_block_name}.norm2.alpha"]))
        layer.norm2.beta = tf.Variable(np.ravel(pt_model_dict[f"{pt_block_name}.norm2.beta"]))


        # ls weights(layerscale)
        layer.ls1.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.ls1"]))

        layer.ls2.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.ls2"]))

    return tf_model