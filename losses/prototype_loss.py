import torch.nn.functional as F


def prototype_loss(similarity, labels):
    return F.cross_entropy(similarity, labels)
