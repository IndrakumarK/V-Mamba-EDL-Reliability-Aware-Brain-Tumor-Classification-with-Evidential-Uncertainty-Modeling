import torch
from models.fusion import decision_fusion


def predict(model, prototype_memory, image, beta=0.5, tau=0.5):
    model.eval()
    with torch.no_grad():
        z, edl_out = model(image)
        similarity = prototype_memory.similarity(z)
        decision = decision_fusion(
            edl_out["belief"], similarity, beta
        )
        uncertainty = edl_out["uncertainty"]

        if uncertainty.item() > tau:
            return "Unknown"
        return torch.argmax(decision, dim=1).item()
