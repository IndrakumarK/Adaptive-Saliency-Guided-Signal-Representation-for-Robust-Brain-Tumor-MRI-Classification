import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from data.dataset import MRIDataset
from models import SaliencyCNN, ASGSRPipeline, BayesianClassifier
from utils import (
    compute_psnr,
    compute_ssim,
    compute_snr,
    plot_confusion_matrix,
    plot_roc_curve,
)


# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
def load_model():
    model = SaliencyCNN(num_classes=config.NUM_CLASSES)

    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model.eval()

    return model


# -------------------------------------------------
# FEATURE EXTRACTION
# -------------------------------------------------
def extract_features(loader, pipeline):
    features = []
    labels = []
    psnr_list = []
    ssim_list = []
    snr_list = []

    for images, y in loader:
        images = images.to(pipeline.device)

        for i in range(images.shape[0]):
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            tensor = images[i].unsqueeze(0)

            feat, saliency = pipeline.process(img, tensor)

            # Signal metrics
            filtered = img * np.expand_dims(saliency, axis=-1)

            psnr_list.append(compute_psnr(img, filtered))
            ssim_list.append(compute_ssim(img, filtered))
            snr_list.append(compute_snr(img, filtered))

            features.append(feat)
            labels.append(y[i].item())

    return (
        np.array(features),
        np.array(labels),
        np.mean(psnr_list),
        np.mean(ssim_list),
        np.mean(snr_list),
    )


# -------------------------------------------------
# MAIN EVALUATION
# -------------------------------------------------
def evaluate():
    device = torch.device(
        config.DEVICE if torch.cuda.is_available() else "cpu"
    )

    # -------------------------------
    # LOAD DATASETS
    # -------------------------------
    train_dataset = MRIDataset(config.DATA_ROOT, split="train", img_size=config.IMG_SIZE)
    test_dataset = MRIDataset(config.DATA_ROOT, split="test", img_size=config.IMG_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # -------------------------------
    # LOAD MODEL
    # -------------------------------
    model = load_model().to(device)

    pipeline = ASGSRPipeline(model, device=device)

    # -------------------------------
    # FEATURE EXTRACTION
    # -------------------------------
    print("Extracting TRAIN features...")
    X_train, y_train, _, _, _ = extract_features(train_loader, pipeline)

    print("Extracting TEST features...")
    X_test, y_test, psnr, ssim, snr = extract_features(test_loader, pipeline)

    # -------------------------------
    # TRAIN CLASSIFIER (CORRECT)
    # -------------------------------
    classifier = BayesianClassifier()
    classifier.fit(X_train, y_train)

    # -------------------------------
    # TESTING
    # -------------------------------
    y_pred, confidence, probs = classifier.predict_with_confidence(X_test)

    accuracy = np.mean(y_pred == y_test)

    # -------------------------------
    # RESULTS
    # -------------------------------
    print("\n===== Evaluation Results =====")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.3f}")
    print(f"SNR: {snr:.2f} dB")
    print(f"Avg Confidence: {np.mean(confidence):.3f}")

    # -------------------------------
    # VISUALIZATION
    # -------------------------------
    class_names = test_dataset.get_class_names()

    plot_confusion_matrix(y_test, y_pred, class_names)
    plot_roc_curve(y_test, probs, config.NUM_CLASSES)


# -------------------------------------------------
# RUN
# -------------------------------------------------
if __name__ == "__main__":
    evaluate()