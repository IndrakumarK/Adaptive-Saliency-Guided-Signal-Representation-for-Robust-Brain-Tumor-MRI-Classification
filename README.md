# Adaptive Saliency-Guided Signal Representation (ASGSR) for Brain Tumor MRI Classification

## 📌 Overview

This repository implements the **Adaptive Saliency-Guided Signal Representation (ASGSR)** framework for robust brain tumor classification using MRI images.

Unlike conventional deep learning approaches, this framework treats MRI images as **structured spatial signals**, integrating:

- Variational saliency estimation  
- Adaptive spatial filtering  
- Multi-resolution signal decomposition  
- Higher-order statistical feature modeling  
- Bayesian decision inference with uncertainty estimation  

The proposed approach improves **robustness, interpretability, and signal fidelity** under noisy and heterogeneous imaging conditions.

---

## 🧠 Key Contributions

- ✔ Signal-centric formulation for MRI analysis  
- ✔ Gradient-based saliency as signal sensitivity estimator  
- ✔ Multi-resolution signal decomposition  
- ✔ Statistical feature representation (mean, variance, skewness, kurtosis)  
- ✔ Bayesian classification with confidence estimation  
- ✔ Robust performance under noise and cross-dataset conditions  

---

## 📁 Project Structure

```

ASGSR-MRI-Classification/
│
├── data/                  # Dataset loaders & preprocessing
├── models/                # CNN, ASGSR pipeline, Bayesian classifier
├── utils/                 # Metrics & visualization
├── train.py               # Training pipeline
├── evaluate.py            # Evaluation script
├── config.py              # Configuration file
├── requirements.txt       # Dependencies


````

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/ASGSR-MRI-Classification.git
cd ASGSR-MRI-Classification

pip install -r requirements.txt
````

Optional (for N4 bias correction):

```bash
pip install SimpleITK
```

---

## 📊 Datasets

The framework is evaluated on multiple publicly available brain MRI datasets:

---

### 🔹 1. Nickparvar Dataset [34]

* 6,920 MRI images
* Classes: Glioma, Meningioma, Pituitary, Normal
* Source: Kaggle

👉 [https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

---

### 🔹 2. Epic & CSCR Dataset (Mendeley) [35]

* 12,064 contrast-enhanced T1 MRI images
* 4 classes with train/test split
* Source: Mendeley Data

👉 [https://doi.org/10.17632/zwr4ntf94j.5](https://doi.org/10.17632/zwr4ntf94j.5)

---

### 🔹 3. BRISC Dataset [36]

* ~6,000 MRI images
* Multi-scanner variability
* Source: Kaggle

👉 [https://www.kaggle.com/datasets/briscdataset/brisc2025/](https://www.kaggle.com/datasets/briscdataset/brisc2025/)

---

### 🔹 4. Figshare Dataset [37]

* 2,980 MRI images
* Classes: Glioma, Meningioma, Pituitary
* Source: Figshare

👉 [https://figshare.com/articles/dataset/brain_tumor_dataset/1512427](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)

---

## 📂 Dataset Structure

```
data/
├── train/
│   ├── Glioma/
│   ├── Meningioma/
│   ├── Pituitary/
│   └── No\_Tumor/
├── test/
│   ├── Glioma/
│   ├── Meningioma/
│   ├── Pituitary/
│   └── No\_Tumor/

```
---

## 🚀 Usage

### 🔹 Train Model

```bash
python train.py
```
---

### 🔹 Evaluate Model

```bash
python evaluate.py
```

---

## 📈 Evaluation Metrics

The framework evaluates both **signal fidelity** and **classification performance**:

### Signal Metrics

* PSNR (Peak Signal-to-Noise Ratio)
* SSIM (Structural Similarity Index)
* SNR (Signal-to-Noise Ratio)

### Classification Metrics

* Accuracy
* Precision / Recall / F1-score
* AUC-ROC
* Matthews Correlation Coefficient (MCC)

### Reliability

* Entropy-based confidence score

---

## 🧪 Experiments

* ✔ Cross-dataset generalization
* ✔ Noise robustness (Gaussian, Rician, Bias field)
* ✔ Ablation studies
* ✔ Signal-level analysis (frequency, entropy, edge preservation)

---

## 📊 Results

| Method         | Accuracy  | AUC      |
| -------------- | --------- | -------- |
| CNN            | 91.3%     | 0.93     |
| Transformer    | 93.1%     | 0.95     |
| Proposed ASGSR | **95.2%** | **0.97** |

---

## 🔬 Key Insight

The ASGSR framework demonstrates that:

> Modeling MRI data as structured signals significantly improves robustness, interpretability, and reliability beyond conventional deep learning approaches.

---

## 📌 Citation

If you use this work, please cite:

```

@article{asg sr2026,
title={Adaptive Saliency-Guided Signal Representation for Robust Brain Tumor MRI Classification},
author={},
journal={},
year={2026}
}

```
---

## 📄 License

This project is licensed under the MIT License.

---

## 🚀 Final Note

This repository provides a **signal-processing-driven alternative to deep learning**, emphasizing:

* Robustness under noise
* Interpretability
* Structured signal modeling

```
