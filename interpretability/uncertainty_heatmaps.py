import torch


def extract_uncertainty_map(uncertainty_tensor):
    u = uncertainty_tensor.detach().cpu()
    u = (u - u.min()) / (u.max() - u.min() + 1e-8)
    return u
