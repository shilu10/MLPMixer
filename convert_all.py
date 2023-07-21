import numpy as np 
import yaml 
import os, sys, shutil, glob
from imutils import paths 
import numpy as np 
import pandas as pd 
from .convert import port 

all_model_types = [
    "mixer_l16_224.goog_in21k_ft_in1k",
    "mixer_b16_224.goog_in21k_ft_in1k",
    "gmixer_12_224",
    "gmixer_24_224",
    "gmlp_b16_224",
    "gmlp_s16_224",
    "gmlp_ti16_224",
    "resmlp_12_224",
    "resmlp_24_224",
    "resmlp_36_224",
    "resmlp_12_224.fb_distilled_in1k",
    "resmlp_24_224.fb_distilled_in1k",
    "resmlp_36_224.fb_distilled_in1k",
    "resmlp_big_24_224.fb_distilled_in1k"
]

def port_all(mtype="classifier", model_savepath="models/"):
    if mtype == "classifier":
        for model_type in all_model_types:
            try: 
                print("Processing model type: ", model_type)
                port(
                    model_type = model_type,
                    model_savepath = model_savepath,
                    include_top = True
                )

            except Exception as err:
                print(err, "some models does'nt have a pretrained weights in pytorch.")
        
    elif mtype == "feature_extractor":
        for model_type in all_model_types:
            try:
                print("Processing model type: ", model_type)
                port(
                    model_type = model_type,
                    model_savepath = ".",
                    include_top = False
                )
            
            except Exception as err:
                print(err, "some models does'nt have a pretrained weights in pytorch.")
    
    
                