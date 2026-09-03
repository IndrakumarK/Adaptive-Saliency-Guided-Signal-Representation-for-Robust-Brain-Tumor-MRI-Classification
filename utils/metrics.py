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

from utils.metrics import (
    compute_psnr,
    compute_ssim,
    compute_snr,
    compute_edge_preservation
)

from utils import (
    plot_confusion_matrix,
    plot_roc_curve
)


# -------------------------------------------------
# LOAD TRAINED SALIENCY CNN
# -------------------------------------------------
def load_model(device):
    model = SaliencyCNN(
        num_classes=config.NUM_CLASSES
    )

    model.load_state_dict(
        torch.load(
            config.MODEL_PATH,
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()

    return model


# -------------------------------------------------
# FEATURE EXTRACTION + SIGNAL METRICS
# -------------------------------------------------
def extract_features(loader, pipeline):

    features = []
    labels = []

    psnr_list = []
    ssim_list = []
    snr_list = []
    edge_list = []

    for images, y in loader:

        images = images.to(
            pipeline.device
        )

        for i in range(images.shape[0]):

            # -------------------------------------
            # ORIGINAL MRI
            # -------------------------------------
            img = (
                images[i]
                .detach()
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
            )

            # Tensor for saliency computation
            tensor = images[i].unsqueeze(0)

            # -------------------------------------
            # ASGSR PROCESSING
            # -------------------------------------
            feat, saliency = pipeline.process(
                img,
                tensor
            )

            # -------------------------------------
            # SALIENCY-GUIDED REPRESENTATION
            # -------------------------------------
            filtered = (
                img
                * np.expand_dims(
                    saliency,
                    axis=-1
                )
            )

            # -------------------------------------
            # SIGNAL-LEVEL METRICS
            # -------------------------------------
            psnr_value = compute_psnr(
                img,
                filtered
            )

            ssim_value = compute_ssim(
                img,
                filtered
            )

            snr_value = compute_snr(
                img,
                filtered
            )

            edge_value = compute_edge_preservation(
                img,
                filtered
            )

            # -------------------------------------
            # STORE METRICS
            # -------------------------------------
            psnr_list.append(psnr_value)
            ssim_list.append(ssim_value)
            snr_list.append(snr_value)
            edge_list.append(edge_value)

            # -------------------------------------
            # STORE FEATURES
            # -------------------------------------
            features.append(feat)
            labels.append(
                y[i].item()
            )

    return (
        np.array(features),
        np.array(labels),
        np.mean(psnr_list),
        np.mean(ssim_list),
        np.mean(snr_list),
        np.mean(edge_list)
    )


# -------------------------------------------------
# MAIN EVALUATION
# -------------------------------------------------
def evaluate():

    # ---------------------------------------------
    # DEVICE
    # ---------------------------------------------
    device = torch.device(
        config.DEVICE
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # ---------------------------------------------
    # DATASETS
    # ---------------------------------------------
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

    # ---------------------------------------------
    # DATA LOADERS
    # ---------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )

    # ---------------------------------------------
    # LOAD TRAINED SALIENCY CNN
    # ---------------------------------------------
    print(
        "\nLoading trained Saliency CNN..."
    )

    model = load_model(device)

    # ---------------------------------------------
    # CREATE ASGSR PIPELINE
    # ---------------------------------------------
    pipeline = ASGSRPipeline(
        model,
        device=device
    )

    # ---------------------------------------------
    # TRAIN FEATURES
    # ---------------------------------------------
    print(
        "\nExtracting TRAIN features..."
    )

    (
        X_train,
        y_train,
        _,
        _,
        _,
        _
    ) = extract_features(
        train_loader,
        pipeline
    )

    print(
        f"Training feature shape: "
        f"{X_train.shape}"
    )

    # ---------------------------------------------
    # TEST FEATURES
    # ---------------------------------------------
    print(
        "\nExtracting TEST features..."
    )

    (
        X_test,
        y_test,
        psnr,
        ssim,
        snr,
        edge_preservation
    ) = extract_features(
        test_loader,
        pipeline
    )

    print(
        f"Testing feature shape: "
        f"{X_test.shape}"
    )

    # ---------------------------------------------
    # BAYESIAN CLASSIFIER
    # ---------------------------------------------
    print(
        "\nTraining Bayesian Classifier..."
    )

    classifier = BayesianClassifier()

    classifier.fit(
        X_train,
        y_train
    )

    # ---------------------------------------------
    # PREDICTION
    # ---------------------------------------------
    y_pred, confidence, probs = (
        classifier.predict_with_confidence(
            X_test
        )
    )

    # ---------------------------------------------
    # CLASSIFICATION ACCURACY
    # ---------------------------------------------
    accuracy = np.mean(
        y_pred == y_test
    )

    # ---------------------------------------------
    # RESULTS
    # ---------------------------------------------
    print(
        "\n===================================="
    )

    print(
        "        ASGSR EVALUATION RESULTS"
    )

    print(
        "===================================="
    )

    print(
        f"Accuracy           : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"PSNR               : "
        f"{psnr:.2f} dB"
    )

    print(
        f"SSIM               : "
        f"{ssim:.3f}"
    )

    print(
        f"SNR                : "
        f"{snr:.2f} dB"
    )

    print(
        f"Edge Preservation  : "
        f"{edge_preservation:.3f}"
    )

    print(
        f"Average Confidence : "
        f"{np.mean(confidence):.3f}"
    )

    print(
        "===================================="
    )

    # ---------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------
    class_names = (
        test_dataset.get_class_names()
    )

    plot_confusion_matrix(
        y_test,
        y_pred,
        class_names
    )

    plot_roc_curve(
        y_test,
        probs,
        config.NUM_CLASSES
    )


# -------------------------------------------------
# RUN
# -------------------------------------------------
if __name__ == "__main__":
    evaluate()
