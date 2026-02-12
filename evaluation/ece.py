import torch


def compute_ece(probs, labels, n_bins=15):
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1)

    for i in range(n_bins):
        mask = (probs > bins[i]) & (probs <= bins[i + 1])
        if mask.sum() > 0:
            acc = (labels[mask] == 1).float().mean()
            conf = probs[mask].mean()
            ece += (mask.float().mean()) * torch.abs(acc - conf)
    return ece.item()
