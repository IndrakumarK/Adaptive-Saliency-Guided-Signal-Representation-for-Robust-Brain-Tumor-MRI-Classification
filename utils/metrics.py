import numpy as np


# -------------------------------------------------
# 1. PSNR (Peak Signal-to-Noise Ratio)
# -------------------------------------------------
def compute_psnr(original, filtered, max_val=1.0):
    """
    Compute PSNR between original and filtered image
    """
    original = original.astype(np.float32)
    filtered = filtered.astype(np.float32)

    mse = np.mean((original - filtered) ** 2)

    if mse == 0:
        return float("inf")

    psnr = 10 * np.log10((max_val ** 2) / (mse + 1e-8))
    return psnr


# -------------------------------------------------
# 2. SSIM (Structural Similarity Index)
# -------------------------------------------------
def compute_ssim(img1, img2):
    """
    Simplified SSIM implementation (global)
    """
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)

    sigma1 = np.var(img1)
    sigma2 = np.var(img2)

    covariance = np.mean((img1 - mu1) * (img2 - mu2))

    ssim = (
        (2 * mu1 * mu2 + C1) * (2 * covariance + C2)
    ) / (
        (mu1 ** 2 + mu2 ** 2 + C1) *
        (sigma1 + sigma2 + C2)
    )

    return ssim


# -------------------------------------------------
# 3. SNR (Signal-to-Noise Ratio)
# -------------------------------------------------
def compute_snr(original, filtered):
    """
    SNR = 10 log10 (||Is||^2 / ||I - Is||^2)
    """
    original = original.astype(np.float32)
    filtered = filtered.astype(np.float32)

    signal_power = np.sum(filtered ** 2)
    noise_power = np.sum((original - filtered) ** 2)

    if noise_power == 0:
        return float("inf")

    snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
    return snr


# -------------------------------------------------
# 4. CONFIDENCE SCORE (Entropy-based)
# -------------------------------------------------
def compute_confidence_score(probs):
    """
    C(f) = 1 - H(P) / log(|C|)
    
    probs: shape (num_classes,)
    """
    probs = np.array(probs)
    eps = 1e-8

    entropy = -np.sum(probs * np.log(probs + eps))
    max_entropy = np.log(len(probs))

    confidence = 1 - (entropy / (max_entropy + eps))
    return confidence


# -------------------------------------------------
# 5. BATCH CONFIDENCE (OPTIONAL)
# -------------------------------------------------
def compute_batch_confidence(probs_batch):
    """
    probs_batch: (N, num_classes)
    """
    return np.array([compute_confidence_score(p) for p in probs_batch])