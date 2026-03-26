import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix


# -------------------------------------------------
# 1. SALIENCY MAP VISUALIZATION
# -------------------------------------------------
def plot_saliency_map(image, saliency, title="Saliency Map"):
    """
    Overlay saliency map on original image
    """
    plt.figure()

    if image.ndim == 3:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap="gray")

    plt.imshow(saliency, cmap="jet", alpha=0.5)
    plt.title(title)
    plt.axis("off")
    plt.show()


# -------------------------------------------------
# 2. ROC CURVE
# -------------------------------------------------
def plot_roc_curve(y_true, y_probs, num_classes):
    """
    Plot ROC curve (multi-class, one-vs-rest)
    """
    plt.figure()

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_probs[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)

    plt.show()


# -------------------------------------------------
# 3. CONFUSION MATRIX
# -------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


# -------------------------------------------------
# 4. COMPARISON BAR PLOT (OPTIONAL)
# -------------------------------------------------
def plot_metric_comparison(methods, values, metric_name="Accuracy"):
    """
    Bar plot for comparing models
    """
    plt.figure()

    plt.bar(methods, values)
    plt.xlabel("Methods")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Comparison")

    plt.xticks(rotation=30)
    plt.grid(True, axis='y')

    plt.show()