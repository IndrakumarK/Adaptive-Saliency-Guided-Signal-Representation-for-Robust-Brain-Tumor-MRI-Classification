"""
Configuration file for ASGSR MRI Classification
"""

# -------------------------------------------------
# DATA CONFIGURATION
# -------------------------------------------------
DATA_ROOT = "./data"          # Root dataset folder
IMG_SIZE = 224                # Input image size
NUM_CLASSES = 4               # Glioma, Meningioma, Pituitary, Normal

# -------------------------------------------------
# TRAINING CONFIGURATION
# -------------------------------------------------
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

DEVICE = "cuda"               # "cuda" or "cpu"
NUM_WORKERS = 4

# -------------------------------------------------
# MODEL CONFIGURATION (Saliency CNN)
# -------------------------------------------------
CNN_CHANNELS = [32, 64, 128, 256]

# -------------------------------------------------
# ASGSR SIGNAL PARAMETERS
# -------------------------------------------------
NUM_RESOLUTION_LEVELS = 3     # K in paper
BETA = 0.1                   # Saliency smoothness weight
LAMBDA_REG = 1e-4            # Regularization (?1)
LAMBDA_CONF = 1e-3           # Confidence penalty (?2)

# -------------------------------------------------
# BAYESIAN CLASSIFIER
# -------------------------------------------------
USE_DIAGONAL_COV = False     # Optional speed optimization

# -------------------------------------------------
# EVALUATION SETTINGS
# -------------------------------------------------
K_FOLDS = 5
USE_CROSS_DATASET = True

# -------------------------------------------------
# NOISE ROBUSTNESS TESTING
# -------------------------------------------------
ENABLE_NOISE_TEST = True

GAUSSIAN_NOISE_STD = [0.01, 0.05, 0.1]
RICIAN_NOISE_STD = [0.01, 0.05, 0.1]
BIAS_FIELD_STRENGTH = [0.1, 0.2]

# -------------------------------------------------
# OUTPUT SETTINGS
# -------------------------------------------------
SAVE_MODEL = True
MODEL_PATH = "./checkpoints/model.pth"

SAVE_FIGURES = True
FIGURE_PATH = "./figures/"

LOG_INTERVAL = 10

# -------------------------------------------------
# RANDOM SEED (REPRODUCIBILITY)
# -------------------------------------------------
SEED = 42