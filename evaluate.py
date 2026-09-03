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


# =================================================
# LOAD TRAINED SALIENCY CNN
# =================================================
def load_model():

    model = SaliencyCNN(
        num_classes=config.NUM_CLASSES
    )

    model.load_state_dict(
        torch.load(
            config.MODEL_PATH,
            map_location=config.DEVICE
        )
    )

    model.eval()

    return model


# =================================================
# FEATURE EXTRACTION AND SIGNAL-LEVEL EVALUATION
# =================================================
def extract_features(loader, pipeline):

    features = []
    labels = []

    psnr_list = []
    ssim_list = []
    snr_list = []

    for images, y in loader:

        images = images.to(
            pipeline.device
        )

        for i in range(
            images.shape[0]
        ):

            # -----------------------------------------
            # Convert tensor to image representation
            # -----------------------------------------
            img = (
                images[i]
                .detach()
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
                .astype(np.float32)
            )

            # Tensor required for gradient computation
            tensor = images[i].unsqueeze(0)

            # -----------------------------------------
            # ASGSR PROCESSING
            # -----------------------------------------
            feat, saliency, filtered = (
                pipeline.process(
                    img,
                    tensor
                )
            )

            # -----------------------------------------
            # SIGNAL-LEVEL EVALUATION
            #
            # The filtered representation returned
            # directly by ASGSR is used.
            # -----------------------------------------
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

            psnr_list.append(
                psnr_value
            )

            ssim_list.append(
                ssim_value
            )

            snr_list.append(
                snr_value
            )

            # -----------------------------------------
            # Store statistical ASGSR features
            # -----------------------------------------
            features.append(
                feat
            )

            labels.append(
                y[i].item()
            )

    return (
        np.asarray(
            features,
            dtype=np.float32
        ),
        np.asarray(
            labels
        ),
        np.mean(psnr_list),
        np.mean(ssim_list),
        np.mean(snr_list)
    )


# =================================================
# MAIN EVALUATION
# =================================================
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

    model = load_model().to(
        device
    )

    # ---------------------------------------------
    # INITIALIZE ASGSR PIPELINE
    # ---------------------------------------------
    pipeline = ASGSRPipeline(
        model,
        device=device
    )

    # =================================================
    # TRAINING REPRESENTATION
    # =================================================
    print(
        "\nExtracting TRAIN ASGSR features..."
    )

    (
        X_train,
        y_train,
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

    # =================================================
    # TEST REPRESENTATION
    # =================================================
    print(
        "\nExtracting TEST ASGSR features..."
    )

    (
        X_test,
        y_test,
        psnr,
        ssim,
        snr
    ) = extract_features(
        test_loader,
        pipeline
    )

    print(
        f"Testing feature shape: "
        f"{X_test.shape}"
    )

    # =================================================
    # BAYESIAN CLASSIFIER
    # =================================================
    print(
        "\nTraining Bayesian classifier..."
    )

    classifier = BayesianClassifier()

    classifier.fit(
        X_train,
        y_train
    )

    # =================================================
    # TEST PREDICTION
    # =================================================
    print(
        "\nPerforming test prediction..."
    )

    (
        y_pred,
        confidence,
        probs
    ) = classifier.predict_with_confidence(
        X_test
    )

    # ---------------------------------------------
    # Classification accuracy
    # ---------------------------------------------
    accuracy = np.mean(
        y_pred == y_test
    )

    # =================================================
    # RESULTS
    # =================================================
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
        f"{psnr:.2f} dB"
    )

    print(
        f"SSIM           : "
        f"{ssim:.3f}"
    )

    print(
        f"SNR            : "
        f"{snr:.2f} dB"
    )

    print(
        f"Avg Confidence : "
        f"{np.mean(confidence):.3f}"
    )

    print(
        "========================================"
    )

    # =================================================
    # CONFUSION MATRIX
    # =================================================
    class_names = (
        test_dataset.get_class_names()
    )

    plot_confusion_matrix(
        y_test,
        y_pred,
        class_names
    )

    # =================================================
    # ROC CURVE
    # =================================================
    plot_roc_curve(
        y_test,
        probs,
        config.NUM_CLASSES
    )


# =================================================
# PROGRAM ENTRY POINT
# =================================================
if __name__ == "__main__":
    evaluate()
