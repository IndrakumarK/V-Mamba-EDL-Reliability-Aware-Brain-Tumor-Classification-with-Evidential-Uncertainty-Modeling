[![Python](https://img.shields.io/badge/python-3.10-blue.svg)]
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange.svg)]
[![License](https://img.shields.io/badge/license-MIT-green.svg)]

# V-Mamba-EDL: Reliability-Aware Brain Tumor Classification with Evidential Uncertainty Modeling

Python • PyTorch • MIT License  

Official implementation of **V-Mamba-EDL** for reliable and uncertainty-aware brain tumor classification from MRI.

This repository implements a reliability-aware framework that integrates state-space representation learning (Vision Mamba), evidential deep learning, and prototype-guided belief regularization to achieve calibrated, risk-aware clinical predictions.

---

## 🧠 Method Overview

V-Mamba-EDL introduces:

- **State-Space Global Representation Learning** via Vision Mamba backbone  
- **Evidential Deep Learning (EDL)** using Dirichlet belief modeling  
- **Prototype-Guided Representation Alignment** for intra-class compactness  
- **Uncertainty-Aware Decision Fusion** for safe abstention  
- **Selective Prediction Mechanism** for clinically reliable deployment  

The framework explicitly models predictive uncertainty rather than relying on deterministic softmax confidence.

---

## 🧩 Architecture Summary

**Input:** 2D MRI slices (T1, T2, FLAIR or grayscale triplicated)  
**Patch Size:** 16 × 16  
**Backbone:** Vision Mamba (state-space modeling)  
**Uncertainty Modeling:** Dirichlet-based evidential learning  
**Prototype Memory Bank:** Class-wise feature centroids  
**Fusion Mechanism:** Belief + cosine similarity  
**Output:** Brain tumor class label or **Unknown** (vacuous state)

---

## 📊 Key Results

- **97.28% Classification Accuracy**
- **2.13% Expected Calibration Error (ECE)**
- **99.34% Accuracy at 20% Uncertainty-Based Rejection**
- Improved cross-dataset robustness
- Linear-complexity state-space modeling for efficient inference

---

## 📁 Repository Structure

V-Mamba-EDL/
│
├── configs/               # Experiment configuration files
├── datasets/              # Dataset loaders
├── evaluation/            # Metrics and calibration evaluation
├── inference/             # Inference scripts
├── interpretability/      # LST maps & uncertainty heatmaps
├── losses/                # Loss functions
├── models/                # Model components
├── robustness/            # Noise robustness experiments
├── training/              # Training pipeline
│
├── environment.yml
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md

---

## 📁 Datasets

- **Brain Tumor MRI (Nickparvar)**  
  https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset  

- **BR35H (Tumor vs No-Tumor)**  
  https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection  

- **Indk214 Dataset**  
  https://www.kaggle.com/datasets/indk214/brain-tumor-dataset  

---

## 🚀 Quick Start

### Install Dependencies

conda env create -f environment.yml
conda activate vmamba_edl

Or:

pip install -r requirements.txt

---

### Train

python training/main.py --config configs/nickparvar.yaml

---

### Inference

python inference/predict.py

---

## 🔬 Reproducibility

- Fixed random seeds  
- Five independent runs  
- Config-controlled hyperparameters  
- Dataset splits provided  
- Deterministic PyTorch settings  

The Vision Mamba block is implemented in a simplified state-space form for reproducibility.

---

## 📄 Citation

@article{Indrakumar2026VMambaEDL,
  title={V-Mamba-EDL: Reliability-Aware Brain Tumor Classification with Evidential Uncertainty Modeling},
  author={Indrakumar K},
  journal={Under Review},
  year={2026}
}

---

## 📜 License

MIT License.
