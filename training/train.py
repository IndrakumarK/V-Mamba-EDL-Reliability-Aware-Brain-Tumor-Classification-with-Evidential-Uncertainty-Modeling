import torch
from torch.optim import Adam
from models.vmamba_edl import VMambaEDL
from models.prototype_memory import PrototypeMemory
from losses.total_loss import total_loss


def train(model, prototype_memory, dataloader, device):
    model.train()
    optimizer = Adam(model.parameters(), lr=1e-4)

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        z, edl_out = model(images)
        prototype_memory.update(z.detach(), labels)

        similarity = prototype_memory.similarity(z)

        target_onehot = torch.nn.functional.one_hot(
            labels, num_classes=similarity.shape[1]
        ).float()

        loss = total_loss(
            edl_out["alpha"],
            target_onehot,
            similarity,
            labels
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
