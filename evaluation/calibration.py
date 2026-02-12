import torch
import numpy as np
import matplotlib.pyplot as plt


def compute_ece(probs, labels, n_bins=15):
    probs = np.array(probs)
    labels = np.array(labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        mask = (confidences > bin_lower) & (confidences <= bin_upper)
        if np.sum(mask) > 0:
            acc = np.mean(predictions[mask] == labels[mask])
            conf = np.mean(confidences[mask])
            ece += (np.sum(mask) / len(probs)) * abs(acc - conf)

    return ece


def plot_reliability_diagram(probs, labels, n_bins=15, save_path=None):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    accs = []
    confs = []

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (
            confidences <= bin_boundaries[i + 1]
        )
        if np.sum(mask) > 0:
            accs.append(np.mean(predictions[mask] == labels[mask]))
            confs.append(np.mean(confidences[mask]))
        else:
            accs.append(0)
            confs.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)

    plt.figure()
    plt.plot(confs, accs, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")

    if save_path:
        plt.savefig(save_path)
    plt.close()
