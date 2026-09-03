import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


# -------------------------------------------------
# HELPER: CONVERT TO GRAYSCALE
# -------------------------------------------------
def _to_grayscale(image):
    """
    Convert an image to grayscale.

    Supports:
        (H, W)
        (H, W, 1)
        (H, W, 3)
    """

    image = np.asarray(image, dtype=np.float32)

    if image.ndim == 2:
        return image

    if image.ndim == 3:

        if image.shape[2] == 1:
            return image[:, :, 0]

        if image.shape[2] == 3:
            return cv2.cvtColor(
                image,
                cv2.COLOR_RGB2GRAY
            )

    raise ValueError(
        f"Unsupported image shape: {image.shape}"
    )


# -------------------------------------------------
# 1. PSNR
# -------------------------------------------------
def compute_psnr(reference, representation):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).

    The reference MRI is compared with the
    derived saliency-guided representation.
    """

    reference = np.asarray(
        reference,
        dtype=np.float32
    )

    representation = np.asarray(
        representation,
        dtype=np.float32
    )

    # Ensure identical shape
    if reference.shape != representation.shape:
        raise ValueError(
            "Reference and representation must "
            "have the same shape."
        )

    data_range = (
        reference.max() - reference.min()
    )

    if data_range <= 1e-8:
        data_range = 1.0

    return peak_signal_noise_ratio(
        reference,
        representation,
        data_range=data_range
    )


# -------------------------------------------------
# 2. SSIM
# -------------------------------------------------
def compute_ssim(reference, representation):
    """
    Compute Structural Similarity Index (SSIM).

    The metric evaluates structural similarity
    between the reference MRI and the derived
    representation.
    """

    reference_gray = _to_grayscale(
        reference
    )

    representation_gray = _to_grayscale(
        representation
    )

    data_range = (
        reference_gray.max()
        - reference_gray.min()
    )

    if data_range <= 1e-8:
        data_range = 1.0

    return structural_similarity(
        reference_gray,
        representation_gray,
        data_range=data_range
    )


# -------------------------------------------------
# 3. SNR
# -------------------------------------------------
def compute_snr(reference, representation):
    """
    Compute Signal-to-Noise Ratio (SNR).

    The residual between the reference MRI and
    the derived representation is treated as
    the signal residual for the evaluation.
    """

    reference = np.asarray(
        reference,
        dtype=np.float32
    )

    representation = np.asarray(
        representation,
        dtype=np.float32
    )

    signal_power = np.mean(
        reference ** 2
    )

    residual = (
        reference - representation
    )

    noise_power = np.mean(
        residual ** 2
    )

    if noise_power <= 1e-12:
        return float("inf")

    snr = 10.0 * np.log10(
        (signal_power + 1e-12)
        / (noise_power + 1e-12)
    )

    return snr


# -------------------------------------------------
# 4. EDGE PRESERVATION
# -------------------------------------------------
def compute_edge_preservation(
    reference,
    representation
):
    """
    Compute edge-preservation similarity.

    Canny edge maps are generated for the
    reference MRI and the derived representation.
    The Dice similarity between the two edge maps
    is reported as the edge-preservation score.

    Range:
        0 -> no edge correspondence
        1 -> complete edge correspondence
    """

    reference_gray = _to_grayscale(
        reference
    )

    representation_gray = _to_grayscale(
        representation
    )

    # ---------------------------------------------
    # Normalize both images to 8-bit
    # ---------------------------------------------
    def normalize_uint8(image):

        image = np.asarray(
            image,
            dtype=np.float32
        )

        min_value = image.min()
        max_value = image.max()

        if max_value - min_value <= 1e-8:
            return np.zeros_like(
                image,
                dtype=np.uint8
            )

        normalized = (
            (image - min_value)
            / (max_value - min_value)
            * 255.0
        )

        return normalized.astype(
            np.uint8
        )

    reference_uint8 = normalize_uint8(
        reference_gray
    )

    representation_uint8 = normalize_uint8(
        representation_gray
    )

    # ---------------------------------------------
    # Canny edge detection
    # ---------------------------------------------
    reference_edges = cv2.Canny(
        reference_uint8,
        threshold1=50,
        threshold2=150
    )

    representation_edges = cv2.Canny(
        representation_uint8,
        threshold1=50,
        threshold2=150
    )

    # Convert to Boolean
    reference_edges = (
        reference_edges > 0
    )

    representation_edges = (
        representation_edges > 0
    )

    # ---------------------------------------------
    # Edge overlap
    # ---------------------------------------------
    intersection = np.logical_and(
        reference_edges,
        representation_edges
    ).sum()

    reference_count = (
        reference_edges.sum()
    )

    representation_count = (
        representation_edges.sum()
    )

    # ---------------------------------------------
    # Dice similarity
    # ---------------------------------------------
    denominator = (
        reference_count
        + representation_count
    )

    if denominator == 0:
        return 1.0

    edge_preservation = (
        2.0 * intersection
        / denominator
    )

    return float(
        edge_preservation
    )
