import torch


def cross_dataset_evaluation(
    model,
    prototype_memory,
    source_loader,
    target_loader,
    device,
):
    model.to(device)
    model.eval()

    results = {}

    for name, loader in [("source", source_loader),
                         ("target", target_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                z, edl_out = model(images)
                similarity = prototype_memory.similarity(z)
                preds = similarity.argmax(dim=1)

                correct += (preds.cpu() == labels).sum().item()
                total += labels.size(0)

        results[name] = correct / total

    return results
