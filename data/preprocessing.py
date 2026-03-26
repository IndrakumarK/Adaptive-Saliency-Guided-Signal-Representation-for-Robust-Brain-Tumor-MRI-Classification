import cv2
import numpy as np

# Optional: SimpleITK for N4 bias correction
try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False


# -------------------------------------------------
# 1. Normalization
# -------------------------------------------------
def normalize_image(image):
    """
    Normalize image to zero mean and unit variance
    """
    image = image.astype(np.float32)
    mean = np.mean(image)
    std = np.std(image) + 1e-8
    return (image - mean) / std


# -------------------------------------------------
# 2. N4 Bias Field Correction (Optional but Strong)
# -------------------------------------------------
def apply_n4_bias_correction(image):
    """
    Apply N4ITK bias field correction (if SimpleITK available)
    """
    if not SITK_AVAILABLE:
        return image  # fallback

    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image

    sitk_image = sitk.GetImageFromArray(image_gray)

    mask_image = sitk.OtsuThreshold(
        sitk_image, 0, 1, 200
    )

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = corrector.Execute(sitk_image, mask_image)

    corrected_np = sitk.GetArrayFromImage(corrected)

    # Convert back to 3-channel if needed
    if len(image.shape) == 3:
        corrected_np = cv2.cvtColor(corrected_np, cv2.COLOR_GRAY2RGB)

    return corrected_np


# -------------------------------------------------
# 3. Skull Stripping (Simple Approximation)
# -------------------------------------------------
def skull_strip(image):
    """
    Basic skull stripping using threshold + masking
    (lightweight alternative to BET/HD-BET)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Otsu threshold
    _, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    mask = mask.astype(np.uint8)

    # Morphological cleaning
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Apply mask
    if len(image.shape) == 3:
        mask = np.stack([mask]*3, axis=-1)

    stripped = image * (mask > 0)

    return stripped


# -------------------------------------------------
# 4. Resize & Standardization
# -------------------------------------------------
def resize_image(image, size=224):
    return cv2.resize(image, (size, size))


# -------------------------------------------------
# 5. Main Preprocessing Pipeline
# -------------------------------------------------
def preprocess_image(image):
    """
    Full preprocessing pipeline:
    - Bias correction
    - Skull stripping
    - Normalization
    """

    # Step 1: Bias correction
    image = apply_n4_bias_correction(image)

    # Step 2: Skull stripping
    image = skull_strip(image)

    # Step 3: Normalize
    image = normalize_image(image)

    return image