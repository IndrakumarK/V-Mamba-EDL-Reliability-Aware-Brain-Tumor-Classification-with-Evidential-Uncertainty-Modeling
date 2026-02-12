import torch
import torch.nn.functional as F


def evidential_loss(alpha, target):
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood = torch.sum(
        target * (torch.digamma(S) - torch.digamma(alpha)),
        dim=1,
        keepdim=True,
    )
    return torch.mean(loglikelihood)
