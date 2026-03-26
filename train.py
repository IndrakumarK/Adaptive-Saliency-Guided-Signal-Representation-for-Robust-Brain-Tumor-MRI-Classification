import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config
from data.dataset import MRIDataset
from models import SaliencyCNN, ASGSRPipeline, BayesianClassifier


# -------------------------------------------------
# TRAIN + VALIDATION LOOP
# -------------------------------------------------
def train_cnn(model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    best_acc = 0.0

    for epoch in range(config.EPOCHS):
        # -------- TRAIN --------
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # -------- VALIDATION --------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total

        print(f"Epoch [{epoch+1}/{config.EPOCHS}] "
              f"Loss: {total_loss:.4f} | Val Acc: {val_acc:.4f}")

        # -------- SAVE BEST MODEL --------
        if val_acc > best_acc:
            best_acc = val_acc

            if config.SAVE_MODEL:
                torch.save(model.state_dict(), config.MODEL_PATH)

    print(f"\nBest Validation Accuracy: {best_acc:.4f}")
    return model


# -------------------------------------------------
# FEATURE EXTRACTION
# -------------------------------------------------
def extract_features(loader, pipeline):
    features = []
    labels = []

    for images, y in loader:
        images = images.to(pipeline.device)

        for i in range(images.shape[0]):
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            tensor = images[i].unsqueeze(0)

            feat, _ = pipeline.process(img, tensor)

            features.append(feat)
            labels.append(y[i].item())

    return np.array(features), np.array(labels)


# -------------------------------------------------
# MAIN TRAINING PIPELINE
# -------------------------------------------------
def train():
    device = torch.device(
        config.DEVICE if torch.cuda.is_available() else "cpu"
    )

    # -------------------------------
    # DATA LOADERS
    # -------------------------------
    train_dataset = MRIDataset(config.DATA_ROOT, split="train", img_size=config.IMG_SIZE)
    val_dataset = MRIDataset(config.DATA_ROOT, split="test", img_size=config.IMG_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # -------------------------------
    # 1. TRAIN SALIENCY CNN
    # -------------------------------
    print("\nTraining Saliency CNN...")
    model = SaliencyCNN(num_classes=config.NUM_CLASSES).to(device)

    model = train_cnn(model, train_loader, val_loader, device)

    print(f"Model saved at: {config.MODEL_PATH}")

    # -------------------------------
    # 2. ASGSR FEATURE EXTRACTION
    # -------------------------------
    print("\nExtracting ASGSR Features...")

    pipeline = ASGSRPipeline(model, device=device)

    X_train, y_train = extract_features(train_loader, pipeline)
    X_val, y_val = extract_features(val_loader, pipeline)

    print(f"Feature shape: {X_train.shape}")

    # -------------------------------
    # 3. TRAIN BAYESIAN CLASSIFIER
    # -------------------------------
    print("\nTraining Bayesian Classifier...")

    classifier = BayesianClassifier()
    classifier.fit(X_train, y_train)

    # Validation performance
    y_pred = classifier.predict(X_val)
    val_acc = np.mean(y_pred == y_val)

    print(f"Bayesian Validation Accuracy: {val_acc:.4f}")

    print("\nTraining completed successfully!")

    return model, classifier


# -------------------------------------------------
# RUN
# -------------------------------------------------
if __name__ == "__main__":
    train()