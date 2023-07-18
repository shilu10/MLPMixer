import numpy as np 
import tensorflow as tf 
from tensorflow.keras import * 
from tensorflow import keras 
from tensorflow.keras.layers import * 
import os, sys, shutil, glob 
import timm 
from .utils import modify_tf_block, get_tf_qkv
from .mlp_mixer.mixer_model import MLPMixer
from .base_config import get_baseconfig
import yaml


def port(model_type, model_savepath, include_top):

    print("Instantiating PyTorch model...")
    pt_model = timm.create_model(
        model_name=model_type, 
        num_classes=1000, 
        pretrained=True
    )

    if "distilled" in model_type:
        assert (
            "dist_token" in pt_model.state_dict()
        ), "Distillation token must be present for models trained with distillation."
    pt_model.eval()

    print("Instantiating TF model...")
    model_cls = MLPMixer

    config_file_path = f'configs/{model_type.replace(".", "_")}.yaml'
    with open(config_file_path, "r") as f:
        data = yaml.safe_load(f)

    config = get_baseconfig(
        model_type = model_type,
        image_size = data.get("image_size"),
        patch_size = data.get("patch_size"),
        depths = data.get("depths"),
        projection_dim = data.get("projection_dim"),
        include_top = include_top,
        mlp_ratio = data.get("mlp_ratio")
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
    tf_model.layers[-3] = modify_tf_block(
        tf_model.layers[-3],
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

    for indx, layer in enumerate(tf_model.layers[1: config.depth+1]):
        pt_block_name = f"blocks.{indx}"
        mlp_name = "mlp_tokens"
        norm_indx = 1
        for inner_block in layer.layers:
            if isinstance(inner_block, FNN):
                inner_block.fc1 = modify_tf_block(
                    inner_block.fc1,
                    pt_model_dict[f"{pt_block_name}.{mlp_name}.fc1.weight"],
                    pt_model_dict[f"{pt_block_name}.{mlp_name}.fc1.bias"]
                )

                inner_block.fc2 = modify_tf_block(
                    inner_block.fc2,
                    pt_model_dict[f"{pt_block_name}.{mlp_name}.fc2.weight"],
                    pt_model_dict[f"{pt_block_name}.{mlp_name}.fc2.bias"]
                )
                mlp_name = "mlp_channels"

            if isinstance(inner_block, layers.Normalization):
                inner_block = modify_tf_block(
                    inner_block,
                    pt_model_dict[f"{pt_block_name}.norm{norm_indx}.weight"],
                    pt_model_dict[f"{pt_block_name}.norm{norm_indx}.bias"]
                )
                norm_indx += 1
        

    print("Porting successful, serializing TensorFlow model...")

    save_path = os.path.join(model_savepath, model_type)
    save_path = f"{save_path}_fe" if not include_top else save_path
    tf_model.save(save_path)
    print(f"TensorFlow model serialized at: {save_path}...")
