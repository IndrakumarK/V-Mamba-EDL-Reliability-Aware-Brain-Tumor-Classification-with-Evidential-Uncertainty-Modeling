import random
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
import torch
import yaml
from torch.utils.data import DataLoader
from models.vmamba_edl import VMambaEDL
from models.prototype_memory import PrototypeMemory
from losses.total_loss import total_loss
from training.trainer import Trainer
from training.ema import EMA
from datasets.brain_mri_dataset import BrainMRIDataset
import random
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VMambaEDL(num_classes=cfg["num_classes"]).to(device)
    prototype_memory = PrototypeMemory(
        cfg["num_classes"], 768
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=1e-4,
    )

    ema = EMA(model)

    trainer = Trainer(
        model,
        prototype_memory,
        optimizer,
        total_loss,
        device,
        ema,
    )

    train_dataset = BrainMRIDataset(
        cfg["train_images"],
        cfg["train_labels"],
        augment=True,
    )

    val_dataset = BrainMRIDataset(
        cfg["val_images"],
        cfg["val_labels"],
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
    )

    for epoch in range(cfg["epochs"]):
        loss = trainer.train_epoch(train_loader)
        metrics = trainer.evaluate(val_loader, cfg["num_classes"])
        print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={metrics['accuracy']:.4f}")

    torch.save(model.state_dict(), "model_final.pth")


if __name__ == "__main__":
    main("configs/default.yaml")
