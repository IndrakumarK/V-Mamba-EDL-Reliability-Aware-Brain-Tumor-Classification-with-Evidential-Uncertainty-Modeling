import torch


def decision_fusion(belief, similarity, beta=0.5):
    """
    Fusion of evidential belief and cosine similarity.

    Cosine similarity ? [-1, 1]
    Belief ? [0, 1]

    To avoid scale mismatch, similarity is normalized to [0, 1].
    """

    # Normalize cosine similarity to [0, 1]
    similarity = (similarity + 1.0) / 2.0

    # Ensure numerical safety
    similarity = torch.clamp(similarity, 0.0, 1.0)

    return beta * belief + (1.0 - beta) * similarity
