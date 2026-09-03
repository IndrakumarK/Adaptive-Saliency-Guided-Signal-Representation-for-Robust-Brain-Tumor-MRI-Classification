```python
"""
Configuration file for ASGSR MRI Classification
"""

# =================================================
# DATA CONFIGURATION
# =================================================

DATA_ROOT = "./data"

# Input MRI image size
IMG_SIZE = 224

# Number of classification categories
# Glioma, Meningioma, Pituitary, Normal
NUM_CLASSES = 4


# =================================================
# TRAINING CONFIGURATION
# =================================================

BATCH_SIZE = 16

EPOCHS = 100

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

DEVICE = "cuda"       # "cuda" or "cpu"

NUM_WORKERS = 4


# =================================================
# MODEL CONFIGURATION
# =================================================

# Number of channels in the Saliency CNN
CNN_CHANNELS = [32, 64, 128, 256]


# =================================================
# ASGSR SIGNAL REPRESENTATION PARAMETERS
# =================================================

# Number of multi-resolution decomposition levels
# K in the manuscript
NUM_RESOLUTION_LEVELS = 4

# -------------------------------------------------
# Saliency smoothness parameter
# -------------------------------------------------
# β controls the contribution of the saliency
# smoothness term in the formulation.
BETA = 0.1

# -------------------------------------------------
# Regularization parameter
# -------------------------------------------------
# λ1 controls the regularization term.
LAMBDA_REG = 1e-4

# -------------------------------------------------
# Confidence penalty parameter
# -------------------------------------------------
# λ2 controls the confidence-related penalty.
LAMBDA_CONF = 1e-3


# =================================================
# BAYESIAN CLASSIFIER CONFIGURATION
# =================================================

# False -> full covariance matrix
# True  -> diagonal covariance matrix
USE_DIAGONAL_COV = False


# =================================================
# EVALUATION CONFIGURATION
# =================================================

# Number of folds for cross-validation
K_FOLDS = 5

# Enable cross-dataset evaluation
USE_CROSS_DATASET = True


# =================================================
# NOISE ROBUSTNESS TESTING
# =================================================

ENABLE_NOISE_TEST = True

# Gaussian noise standard deviations
GAUSSIAN_NOISE_STD = [
    0.01,
    0.05,
    0.10
]

# Rician noise standard deviations
RICIAN_NOISE_STD = [
    0.01,
    0.05,
    0.10
]

# Bias-field strengths
BIAS_FIELD_STRENGTH = [
    0.10,
    0.20
]


# =================================================
# OUTPUT CONFIGURATION
# =================================================

SAVE_MODEL = True

MODEL_PATH = "./checkpoints/model.pth"

SAVE_FIGURES = True

FIGURE_PATH = "./figures/"

LOG_INTERVAL = 10


# =================================================
# REPRODUCIBILITY
# =================================================

SEED = 42
```
