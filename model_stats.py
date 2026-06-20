import torch
from thop import profile
from models.tinyunet3d import TinyUNet3D

def compute_model_stats(model, input_shape=(1,4,128,128,128), device="cuda"):
    model = model.to(device)
    dummy = torch.randn(*input_shape).to(device)

    with torch.no_grad():
        flops, params = profile(model, inputs=(dummy,), verbose=False)

    flops_g = flops / 1e9
    params_m = params / 1e6

    print(f"Params: {params_m:.2f} M")
    print(f"FLOPs : {flops_g:.2f} G")

    return params_m, flops_g

