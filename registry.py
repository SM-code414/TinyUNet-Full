from models.unet3d import UNet3D
from models.tinyunet3d import TinyUNet3D
from models.attentionunet3d import AttentionUNet3D
from models.resunet3d import ResUNet3D
from models.tinyunet_progressive_ablation import (
    TinyUNet_Baseline,
    TinyUNet_Edge,
    TinyUNet_Edge_DS,
    TinyUNet_Full,
)

MODEL_REGISTRY = {

    # original models
    "unet": UNet3D,
    "tinyunet": TinyUNet3D,
    "attentionunet": AttentionUNet3D,
    "resunet": ResUNet3D,

    # manuscript ablations
    "tinyunet_baseline": TinyUNet_Baseline,
    "tinyunet_edge": TinyUNet_Edge,
    "tinyunet_edge_ds": TinyUNet_Edge_DS,
    "tinyunet_full": TinyUNet_Full,
}


def get_model(name, **kwargs):

    name = name.lower()

    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name}")

    return MODEL_REGISTRY[name](**kwargs)