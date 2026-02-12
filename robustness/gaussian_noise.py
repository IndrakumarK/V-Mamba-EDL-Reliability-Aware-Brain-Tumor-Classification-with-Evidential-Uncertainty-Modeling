import torch


def add_gaussian_noise(image, sigma=0.03):
    noise = torch.randn_like(image) * sigma
    return torch.clamp(image + noise, 0, 1)
