import torch
import numpy as np
from tqdm import tqdm

from evaluation.metrics import compute_classification_metrics
from evaluation.calibration import compute_ece
from models.fusion import decision_fusion


class Trainer:
    """
    Trainer class for V-Mamba-EDL.

    Handles:
    - Training loop
    - Evaluation
    - EMA updates
    - Dirichlet predictive probability computation
    - Calibration metrics
    """

    def __init__(
        self,
        model,
        prototype_memory,
        optimizer,
        loss_fn,
        device,
        beta=0.5,
        tau=0.5,
        ema=None,
    ):
        self.model = model
        self.prototype_memory = prototype_memory
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.beta = beta
        self.tau = tau
        self.ema = ema

    # ==========================================================
    # TRAINING
    # ==========================================================

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0

        loop = tqdm(dataloader, desc="Training", leave=False)

        for images, labels in loop:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            z, edl_out = self.model(images)

            # Update prototype memory
            self.prototype_memory.update(z.detach(), labels)

            # Cosine similarity
            similarity = self.prototype_memory.similarity(z)

            # One-hot targets
            target_onehot = torch.nn.functional.one_hot(
                labels, num_classes=similarity.shape[1]
            ).float()

            # Compute total loss
            loss = self.loss_fn(
                edl_out["alpha"],
                target_onehot,
                similarity,
                labels,
            )

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # EMA update
            if self.ema:
                self.ema.update()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        return total_loss / len(dataloader)

    # ==========================================================
    # EVALUATION
    # ==========================================================

    def evaluate(self, dataloader, num_classes):
        """
        Uses Dirichlet predictive mean:
        p_k = alpha_k / S
        """

        self.model.eval()

        y_true = []
        y_pred = []
        y_prob = []
        y_uncertainty = []

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
                images = images.to(self.device)

                z, edl_out = self.model(images)

                alpha = edl_out["alpha"]
                S = torch.sum(alpha, dim=1, keepdim=True)

                # Proper Dirichlet predictive probability
                probs = alpha / S

                # Store uncertainty
                uncertainty = edl_out["uncertainty"]

                preds = torch.argmax(probs, dim=1)

                y_true.extend(labels.numpy())
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())
                y_uncertainty.extend(uncertainty.cpu().numpy())

        # Convert to numpy
        y_prob = np.array(y_prob)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_uncertainty = np.array(y_uncertainty)

        # Compute classification metrics
        metrics = compute_classification_metrics(
            y_true, y_pred, y_prob, num_classes
        )

        # Compute calibration
        metrics["ece"] = compute_ece(y_prob, y_true)

        # Mean uncertainty
        metrics["mean_uncertainty"] = float(np.mean(y_uncertainty))

        return metrics

    # ==========================================================
    # SELECTIVE PREDICTION (UNCERTAINTY-BASED REJECTION)
    # ==========================================================

    def selective_prediction(self, dataloader, num_classes):
        """
        Computes retained accuracy under uncertainty-based rejection.
        """

        self.model.eval()

        all_probs = []
        all_labels = []
        all_uncertainty = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)

                z, edl_out = self.model(images)

                alpha = edl_out["alpha"]
                S = torch.sum(alpha, dim=1, keepdim=True)
                probs = alpha / S

                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_uncertainty.extend(edl_out["uncertainty"].cpu().numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        all_uncertainty = np.array(all_uncertainty)

        # Sort by uncertainty (low to high confidence)
        sorted_indices = np.argsort(all_uncertainty)

        results = {}

        for rejection_rate in [0.0, 0.05, 0.10, 0.20]:
            cutoff = int(len(sorted_indices) * (1 - rejection_rate))
            selected_idx = sorted_indices[:cutoff]

            selected_probs = all_probs[selected_idx]
            selected_labels = all_labels[selected_idx]
            selected_preds = np.argmax(selected_probs, axis=1)

            accuracy = np.mean(selected_preds == selected_labels)
            results[rejection_rate] = accuracy

        return results
