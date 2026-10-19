import numpy as np
import torch
from torch.utils.data import DataLoader

import config

from data.dataset import MRIDataset

from models import (
    SaliencyCNN,
    ASGSRPipeline,
    BayesianClassifier
)

from utils import (
    compute_psnr,
    compute_ssim,
    compute_snr,
    plot_confusion_matrix,
    plot_roc_curve,
)


# =========================================================
# LOAD TRAINED SALIENCY CNN
# =========================================================

def load_model(device):
    """
    Load the trained SaliencyCNN.

    Parameters
    ----------
    device : torch.device
        Device used for inference.

    Returns
    -------
    torch.nn.Module
        Trained SaliencyCNN.
    """

    model = SaliencyCNN(
        num_classes=config.NUM_CLASSES
    )

    checkpoint = torch.load(
        config.MODEL_PATH,
        map_location=device
    )

    # -----------------------------------------------------
    # Handle both plain state_dict and checkpoint formats
    # -----------------------------------------------------

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # -----------------------------------------------------
    # Remove DataParallel prefix if present
    # -----------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model = model.to(device)

    model.eval()

    return model


# =========================================================
# SIGNAL-LEVEL METRIC HELPER
# =========================================================

def compute_signal_metrics(
    original,
    filtered
):
    """
    Compute signal-level similarity metrics.

    Parameters
    ----------
    original : np.ndarray
        Original MRI representation.

    filtered : np.ndarray
        Saliency-weighted MRI representation.

    Returns
    -------
    tuple
        PSNR, SSIM and SNR.
    """

    psnr_value = compute_psnr(
        original,
        filtered
    )

    ssim_value = compute_ssim(
        original,
        filtered
    )

    snr_value = compute_snr(
        original,
        filtered
    )

    return (
        float(psnr_value),
        float(ssim_value),
        float(snr_value)
    )


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(
    loader,
    pipeline
):
    """
    Extract ASGSR statistical representations.

    Processing:
        MRI
          ↓
        Predicted-class saliency
          ↓
        Saliency-guided weighting
          ↓
        Multi-resolution decomposition
          ↓
        Statistical descriptors

    Parameters
    ----------
    loader : DataLoader
        MRI dataset loader.

    pipeline : ASGSRPipeline
        Initialized ASGSR pipeline.

    Returns
    -------
    X : np.ndarray
        ASGSR feature matrix.

    y : np.ndarray
        Ground-truth labels.

    psnr : float
        Mean PSNR.

    ssim : float
        Mean SSIM.

    snr : float
        Mean SNR.
    """

    features = []
    labels = []

    psnr_list = []
    ssim_list = []
    snr_list = []

    # -----------------------------------------------------
    # Disable parameter-gradient accumulation.
    #
    # Saliency itself still requires an input gradient.
    # The ASGSR pipeline explicitly enables it internally.
    # -----------------------------------------------------

    for images, y in loader:

        images = images.to(
            pipeline.device,
            non_blocking=True
        )

        batch_size = images.shape[0]

        for i in range(batch_size):

            # =================================================
            # ORIGINAL IMAGE
            # =================================================

            tensor = images[
                i
            ].unsqueeze(0)

            img = (
                images[i]
                .detach()
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
                .astype(np.float32)
            )

            # =================================================
            # ASGSR PROCESSING
            # =================================================

            result = pipeline.process(
                img,
                tensor
            )

            # -------------------------------------------------
            # Current pipeline returns:
            #
            #     features, saliency
            #
            # -------------------------------------------------

            feat, saliency = result

            # =================================================
            # RECONSTRUCT SALIENCY-WEIGHTED REPRESENTATION
            # =================================================

            filtered = pipeline.apply_saliency_filter(
                img,
                saliency
            )

            # =================================================
            # SIGNAL-LEVEL EVALUATION
            # =================================================

            psnr_value, ssim_value, snr_value = (
                compute_signal_metrics(
                    img,
                    filtered
                )
            )

            psnr_list.append(
                psnr_value
            )

            ssim_list.append(
                ssim_value
            )

            snr_list.append(
                snr_value
            )

            # =================================================
            # STORE ASGSR FEATURES
            # =================================================

            features.append(
                feat
            )

            labels.append(
                int(y[i].item())
            )

    # =========================================================
    # CONVERT TO ARRAYS
    # =========================================================

    X = np.asarray(
        features,
        dtype=np.float32
    )

    y = np.asarray(
        labels,
        dtype=np.int64
    )

    # ---------------------------------------------------------
    # Check for invalid feature values
    # ---------------------------------------------------------

    if not np.all(
        np.isfinite(X)
    ):
        raise ValueError(
            "Non-finite values detected in the ASGSR "
            "feature matrix."
        )

    # ---------------------------------------------------------
    # Mean signal-level metrics
    # ---------------------------------------------------------

    mean_psnr = (
        float(np.mean(psnr_list))
        if len(psnr_list) > 0
        else 0.0
    )

    mean_ssim = (
        float(np.mean(ssim_list))
        if len(ssim_list) > 0
        else 0.0
    )

    mean_snr = (
        float(np.mean(snr_list))
        if len(snr_list) > 0
        else 0.0
    )

    return (
        X,
        y,
        mean_psnr,
        mean_ssim,
        mean_snr
    )


# =========================================================
# BAYESIAN CLASSIFIER TRAINING
# =========================================================

def train_bayesian_classifier(
    X_train,
    y_train
):
    """
    Train the Gaussian Bayesian classifier
    using ASGSR statistical features.
    """

    classifier = BayesianClassifier()

    classifier.fit(
        X_train,
        y_train
    )

    return classifier


# =========================================================
# TEST PREDICTION
# =========================================================

def predict_test_set(
    classifier,
    X_test
):
    """
    Perform Bayesian prediction and entropy-based
    confidence estimation.
    """

    (
        y_pred,
        confidence,
        probabilities
    ) = classifier.predict_with_confidence(
        X_test
    )

    return (
        y_pred,
        confidence,
        probabilities
    )


# =========================================================
# CLASSIFICATION ACCURACY
# =========================================================

def compute_accuracy(
    y_true,
    y_pred
):
    """
    Compute classification accuracy.
    """

    if len(y_true) == 0:
        return 0.0

    return float(
        np.mean(
            y_pred == y_true
        )
    )


# =========================================================
# MAIN EVALUATION
# =========================================================

def evaluate():

    print(
        "\n========================================"
    )

    print(
        "        ASGSR MRI EVALUATION"
    )

    print(
        "========================================"
    )

    # =====================================================
    # DEVICE
    # =====================================================

    if (
        str(config.DEVICE).startswith("cuda")
        and torch.cuda.is_available()
    ):
        device = torch.device(
            config.DEVICE
        )
    else:
        device = torch.device(
            "cpu"
        )

    print(
        f"\nUsing device: {device}"
    )

    # =====================================================
    # DATASETS
    # =====================================================

    print(
        "\nLoading datasets..."
    )

    train_dataset = MRIDataset(
        config.DATA_ROOT,
        split="train",
        img_size=config.IMG_SIZE
    )

    test_dataset = MRIDataset(
        config.DATA_ROOT,
        split="test",
        img_size=config.IMG_SIZE
    )

    print(
        f"Training samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Testing samples : "
        f"{len(test_dataset)}"
    )

    # =====================================================
    # DATA LOADERS
    # =====================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        )
    )

    # =====================================================
    # LOAD TRAINED SALIENCY CNN
    # =====================================================

    print(
        "\nLoading trained Saliency CNN..."
    )

    model = load_model(
        device
    )

    print(
        "Saliency CNN loaded successfully."
    )

    # =====================================================
    # INITIALIZE ASGSR PIPELINE
    # =====================================================

    print(
        "\nInitializing ASGSR pipeline..."
    )

    pipeline = ASGSRPipeline(
        model=model,
        device=device,
        num_levels=4
    )

    print(
        "Multi-resolution levels: 4"
    )

    # =====================================================
    # TRAINING ASGSR REPRESENTATION
    # =====================================================

    print(
        "\n----------------------------------------"
    )

    print(
        "Extracting TRAIN ASGSR features..."
    )

    print(
        "----------------------------------------"
    )

    (
        X_train,
        y_train,
        train_psnr,
        train_ssim,
        train_snr
    ) = extract_features(
        train_loader,
        pipeline
    )

    print(
        f"Training feature shape: "
        f"{X_train.shape}"
    )

    print(
        f"Training PSNR: "
        f"{train_psnr:.2f} dB"
    )

    print(
        f"Training SSIM: "
        f"{train_ssim:.3f}"
    )

    print(
        f"Training SNR : "
        f"{train_snr:.2f} dB"
    )

    # =====================================================
    # TEST ASGSR REPRESENTATION
    # =====================================================

    print(
        "\n----------------------------------------"
    )

    print(
        "Extracting TEST ASGSR features..."
    )

    print(
        "----------------------------------------"
    )

    (
        X_test,
        y_test,
        test_psnr,
        test_ssim,
        test_snr
    ) = extract_features(
        test_loader,
        pipeline
    )

    print(
        f"Testing feature shape: "
        f"{X_test.shape}"
    )

    # =====================================================
    # BAYESIAN CLASSIFIER
    # =====================================================

    print(
        "\n----------------------------------------"
    )

    print(
        "Training Bayesian classifier..."
    )

    print(
        "----------------------------------------"
    )

    classifier = train_bayesian_classifier(
        X_train,
        y_train
    )

    print(
        "Bayesian classifier trained successfully."
    )

    # =====================================================
    # TEST PREDICTION
    # =====================================================

    print(
        "\n----------------------------------------"
    )

    print(
        "Performing Bayesian test prediction..."
    )

    print(
        "----------------------------------------"
    )

    (
        y_pred,
        confidence,
        probs
    ) = predict_test_set(
        classifier,
        X_test
    )

    # =====================================================
    # CLASSIFICATION ACCURACY
    # =====================================================

    accuracy = compute_accuracy(
        y_test,
        y_pred
    )

    mean_confidence = float(
        np.mean(
            confidence
        )
    )

    # =====================================================
    # RESULTS
    # =====================================================

    print(
        "\n========================================"
    )

    print(
        "       ASGSR EVALUATION RESULTS"
    )

    print(
        "========================================"
    )

    print(
        f"Accuracy       : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"PSNR           : "
        f"{test_psnr:.2f} dB"
    )

    print(
        f"SSIM           : "
        f"{test_ssim:.3f}"
    )

    print(
        f"SNR            : "
        f"{test_snr:.2f} dB"
    )

    print(
        f"Avg Confidence : "
        f"{mean_confidence:.3f}"
    )

    print(
        f"Feature dim.   : "
        f"{X_test.shape[1]}"
    )

    print(
        "========================================"
    )

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    print(
        "\nGenerating confusion matrix..."
    )

    class_names = (
        test_dataset.get_class_names()
    )

    plot_confusion_matrix(
        y_test,
        y_pred,
        class_names
    )

    # =====================================================
    # ROC CURVE
    # =====================================================

    print(
        "Generating ROC curve..."
    )

    plot_roc_curve(
        y_test,
        probs,
        config.NUM_CLASSES
    )

    print(
        "\nEvaluation completed successfully."
    )


# =========================================================
# PROGRAM ENTRY POINT
# =========================================================

if __name__ == "__main__":

    evaluate()
