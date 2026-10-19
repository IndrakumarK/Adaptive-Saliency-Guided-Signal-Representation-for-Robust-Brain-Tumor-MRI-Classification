import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config

from data.dataset import MRIDataset

from models import (
    SaliencyCNN,
    ASGSRPipeline,
    BayesianClassifier,
)


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_seed(seed):
    """
    Set random seeds for reproducible training.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(
            seed
        )

        torch.cuda.manual_seed_all(
            seed
        )

    # Deterministic behavior.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# MODEL CHECKPOINT
# =========================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_acc,
    path
):
    """
    Save the best CNN checkpoint.

    The checkpoint contains the model state and
    optimization state for reproducibility.
    """

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_accuracy": val_acc,
        },
        path
    )


# =========================================================
# LOAD CHECKPOINT
# =========================================================

def load_checkpoint(
    model,
    path,
    device
):
    """
    Load a previously saved model checkpoint.

    Supports both:
        1. Full checkpoint dictionaries
        2. Plain state_dict files
    """

    checkpoint = torch.load(
        path,
        map_location=device
    )

    if isinstance(
        checkpoint,
        dict
    ):

        if "model_state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )

            return checkpoint

        if "state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint[
                    "state_dict"
                ]
            )

            return checkpoint

    # Plain state_dict
    model.load_state_dict(
        checkpoint
    )

    return {
        "epoch": 0,
        "val_accuracy": 0.0
    }


# =========================================================
# VALIDATION
# =========================================================

def validate_cnn(
    model,
    val_loader,
    device,
    criterion
):
    """
    Evaluate the SaliencyCNN on the validation set.

    Returns
    -------
    val_loss : float
    val_accuracy : float
    """

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            batch_size = labels.size(
                0
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += batch_size

    if total == 0:
        return 0.0, 0.0

    val_loss = (
        total_loss / total
    )

    val_accuracy = (
        correct / total
    )

    return (
        val_loss,
        val_accuracy
    )


# =========================================================
# TRAIN SALIENCY CNN
# =========================================================

def train_cnn(
    model,
    train_loader,
    val_loader,
    device
):
    """
    Train the SaliencyCNN.

    This function deliberately keeps the model-specific
    saliency objective inside SaliencyCNN. The current
    training script cannot safely invent additional
    manuscript-specific loss terms without the actual
    SaliencyCNN implementation.

    Optimization:
        Adam
        learning rate = config.LEARNING_RATE
        weight decay  = config.WEIGHT_DECAY
    """

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    best_accuracy = -np.inf

    best_epoch = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    # -----------------------------------------------------
    # Ensure checkpoint directory exists
    # -----------------------------------------------------

    checkpoint_directory = os.path.dirname(
        config.MODEL_PATH
    )

    if checkpoint_directory:
        os.makedirs(
            checkpoint_directory,
            exist_ok=True
        )

    # =====================================================
    # EPOCH LOOP
    # =====================================================

    for epoch in range(
        config.EPOCHS
    ):

        # -------------------------------------------------
        # TRAINING
        # -------------------------------------------------

        model.train()

        running_loss = 0.0
        training_samples = 0

        for images, labels in train_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            batch_size = labels.size(
                0
            )

            running_loss += (
                loss.item()
                * batch_size
            )

            training_samples += (
                batch_size
            )

        if training_samples > 0:

            train_loss = (
                running_loss /
                training_samples
            )

        else:

            train_loss = 0.0

        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------

        val_loss, val_accuracy = (
            validate_cnn(
                model,
                val_loader,
                device,
                criterion
            )
        )

        # -------------------------------------------------
        # Store history
        # -------------------------------------------------

        history[
            "train_loss"
        ].append(
            train_loss
        )

        history[
            "val_loss"
        ].append(
            val_loss
        )

        history[
            "val_accuracy"
        ].append(
            val_accuracy
        )

        # -------------------------------------------------
        # Display
        # -------------------------------------------------

        print(
            f"Epoch "
            f"[{epoch + 1:03d}/"
            f"{config.EPOCHS:03d}] "
            f"Train Loss: "
            f"{train_loss:.6f} | "
            f"Val Loss: "
            f"{val_loss:.6f} | "
            f"Val Acc: "
            f"{val_accuracy:.4f}"
        )

        # -------------------------------------------------
        # Save best model
        # -------------------------------------------------

        if (
            val_accuracy >
            best_accuracy
        ):

            best_accuracy = (
                val_accuracy
            )

            best_epoch = (
                epoch + 1
            )

            if config.SAVE_MODEL:

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    val_acc=val_accuracy,
                    path=config.MODEL_PATH
                )

                print(
                    "  -> Best model saved."
                )

    # =====================================================
    # LOAD BEST MODEL
    # =====================================================

    if (
        config.SAVE_MODEL
        and os.path.exists(
            config.MODEL_PATH
        )
    ):

        print(
            "\nLoading best CNN checkpoint..."
        )

        load_checkpoint(
            model,
            config.MODEL_PATH,
            device
        )

    print(
        "\nBest Validation Accuracy: "
        f"{best_accuracy:.4f}"
    )

    print(
        f"Best Epoch: {best_epoch}"
    )

    return (
        model,
        history
    )


# =========================================================
# ASGSR FEATURE EXTRACTION
# =========================================================

def extract_features(
    loader,
    pipeline
):
    """
    Extract ASGSR statistical representations.

    Pipeline:
        MRI
          ↓
        Predicted-class gradient saliency
          ↓
        Saliency-guided weighting
          ↓
        Multi-resolution decomposition
          ↓
        Statistical descriptors
    """

    features = []
    labels = []

    pipeline.model.eval()

    for images, y in loader:

        images = images.to(
            pipeline.device,
            non_blocking=True
        )

        for i in range(
            images.shape[0]
        ):

            # -------------------------------------------------
            # Individual MRI tensor
            # -------------------------------------------------

            tensor = images[
                i
            ].unsqueeze(0)

            # -------------------------------------------------
            # Convert to H x W x C representation
            # -------------------------------------------------

            image = (
                images[i]
                .detach()
                .cpu()
                .numpy()
                .transpose(
                    1,
                    2,
                    0
                )
                .astype(
                    np.float32
                )
            )

            # -------------------------------------------------
            # ASGSR processing
            # -------------------------------------------------

            feature_vector, saliency = (
                pipeline.process(
                    image,
                    tensor
                )
            )

            # -------------------------------------------------
            # Validate feature vector
            # -------------------------------------------------

            feature_vector = np.asarray(
                feature_vector,
                dtype=np.float32
            )

            if not np.all(
                np.isfinite(
                    feature_vector
                )
            ):
                raise ValueError(
                    "Non-finite values detected "
                    "in ASGSR features."
                )

            features.append(
                feature_vector
            )

            labels.append(
                int(
                    y[i].item()
                )
            )

    X = np.asarray(
        features,
        dtype=np.float32
    )

    y = np.asarray(
        labels,
        dtype=np.int64
    )

    return (
        X,
        y
    )


# =========================================================
# BAYESIAN CLASSIFIER
# =========================================================

def train_bayesian_classifier(
    X_train,
    y_train
):
    """
    Fit the Gaussian Bayesian classifier on
    ASGSR statistical features.
    """

    classifier = (
        BayesianClassifier()
    )

    classifier.fit(
        X_train,
        y_train
    )

    return classifier


# =========================================================
# BAYESIAN VALIDATION
# =========================================================

def evaluate_bayesian_classifier(
    classifier,
    X,
    y
):
    """
    Evaluate Bayesian classification accuracy.
    """

    y_pred = classifier.predict(
        X
    )

    accuracy = float(
        np.mean(
            y_pred == y
        )
    )

    return (
        accuracy,
        y_pred
    )


# =========================================================
# MAIN TRAINING PIPELINE
# =========================================================

def train():

    print(
        "\n========================================"
    )

    print(
        "       ASGSR TRAINING PIPELINE"
    )

    print(
        "========================================"
    )

    # =====================================================
    # REPRODUCIBILITY
    # =====================================================

    set_seed(
        config.SEED
    )

    print(
        f"\nRandom seed: "
        f"{config.SEED}"
    )

    # =====================================================
    # DEVICE
    # =====================================================

    if (
        str(config.DEVICE).startswith(
            "cuda"
        )
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
        f"Device: {device}"
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

    val_dataset = MRIDataset(
        config.DATA_ROOT,
        split="test",
        img_size=config.IMG_SIZE
    )

    print(
        f"Training samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    # =====================================================
    # DATA LOADERS
    # =====================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        )
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        )
    )

    # =====================================================
    # 1. TRAIN SALIENCY CNN
    # =====================================================

    print(
        "\n========================================"
    )

    print(
        "1. TRAINING SALIENCY CNN"
    )

    print(
        "========================================"
    )

    model = SaliencyCNN(
        num_classes=config.NUM_CLASSES
    ).to(
        device
    )

    model, history = train_cnn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    if config.SAVE_MODEL:

        print(
            f"\nBest model saved at:"
            f"\n{config.MODEL_PATH}"
        )

    # =====================================================
    # 2. INITIALIZE ASGSR
    # =====================================================

    print(
        "\n========================================"
    )

    print(
        "2. INITIALIZING ASGSR"
    )

    print(
        "========================================"
    )

    pipeline = ASGSRPipeline(
        model=model,
        device=device,
        num_levels=4
    )

    # =====================================================
    # 3. TRAIN ASGSR REPRESENTATION
    # =====================================================

    print(
        "\n========================================"
    )

    print(
        "3. EXTRACTING TRAIN ASGSR FEATURES"
    )

    print(
        "========================================"
    )

    (
        X_train,
        y_train
    ) = extract_features(
        train_loader,
        pipeline
    )

    print(
        f"Training feature matrix: "
        f"{X_train.shape}"
    )

    # =====================================================
    # 4. VALIDATION ASGSR REPRESENTATION
    # =====================================================

    print(
        "\n========================================"
    )

    print(
        "4. EXTRACTING VALIDATION FEATURES"
    )

    print(
        "========================================"
    )

    (
        X_val,
        y_val
    ) = extract_features(
        val_loader,
        pipeline
    )

    print(
        f"Validation feature matrix: "
        f"{X_val.shape}"
    )

    # =====================================================
    # 5. TRAIN BAYESIAN CLASSIFIER
    # =====================================================

    print(
        "\n========================================"
    )

    print(
        "5. TRAINING BAYESIAN CLASSIFIER"
    )

    print(
        "========================================"
    )

    classifier = (
        train_bayesian_classifier(
            X_train,
            y_train
        )
    )

    print(
        "Bayesian classifier fitted successfully."
    )

    # =====================================================
    # 6. BAYESIAN VALIDATION
    # =====================================================

    print(
        "\n========================================"
    )

    print(
        "6. BAYESIAN VALIDATION"
    )

    print(
        "========================================"
    )

    (
        val_accuracy,
        y_pred
    ) = evaluate_bayesian_classifier(
        classifier,
        X_val,
        y_val
    )

    print(
        f"Bayesian Validation Accuracy: "
        f"{val_accuracy:.4f}"
    )

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print(
        "\n========================================"
    )

    print(
        "       TRAINING COMPLETED"
    )

    print(
        "========================================"
    )

    print(
        f"Saliency CNN best validation accuracy: "
        f"{max(history['val_accuracy']):.4f}"
    )

    print(
        f"ASGSR feature dimension: "
        f"{X_train.shape[1]}"
    )

    print(
        f"Bayesian validation accuracy: "
        f"{val_accuracy:.4f}"
    )

    print(
        "========================================"
    )

    return (
        model,
        classifier
    )


# =========================================================
# PROGRAM ENTRY POINT
# =========================================================

if __name__ == "__main__":

    train()
