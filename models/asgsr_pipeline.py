import numpy as np
import cv2
import torch
import torch.nn.functional as F


class ASGSRPipeline:
    """
    Adaptive Saliency-Guided Signal Representation (ASGSR)

    Pipeline:
    1. Saliency estimation (gradient-based)
    2. Adaptive filtering
    3. Multi-resolution decomposition
    4. Statistical feature extraction
    """

    def __init__(self, model, device="cpu", num_levels=3):
        """
        model: Saliency CNN (used for gradient computation)
        """
        self.model = model
        self.device = device
        self.num_levels = num_levels

        self.model.to(self.device)
        self.model.eval()

    # -------------------------------------------------
    # 1. SALIENCY ESTIMATION
    # -------------------------------------------------
    def compute_saliency(self, image_tensor):
        """
        Compute gradient-based saliency map
        image_tensor: shape (1, C, H, W)
        """
        image_tensor = image_tensor.clone().detach().to(self.device)
        image_tensor.requires_grad = True

        output = self.model(image_tensor)
        score, _ = torch.max(output, dim=1)

        self.model.zero_grad()
        score.backward()

        saliency = image_tensor.grad.data.abs()

        # Aggregate across channels
        saliency, _ = torch.max(saliency, dim=1)
        saliency = saliency.squeeze().cpu().numpy()

        # Normalize to [0,1]
        saliency = saliency / (saliency.max() + 1e-8)

        return saliency

    # -------------------------------------------------
    # 2. ADAPTIVE FILTERING
    # -------------------------------------------------
    def apply_saliency_filter(self, image, saliency):
        """
        Element-wise multiplication
        """
        if len(image.shape) == 3:
            saliency = np.expand_dims(saliency, axis=-1)

        filtered = image * saliency
        return filtered

    # -------------------------------------------------
    # 3. MULTI-RESOLUTION DECOMPOSITION
    # -------------------------------------------------
    def multi_resolution_decomposition(self, image):
        """
        Simple pyramid decomposition
        """
        components = []
        current = image.copy()

        for _ in range(self.num_levels):
            down = cv2.pyrDown(current)
            up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))

            laplacian = current - up
            components.append(laplacian)

            current = down

        components.append(current)  # lowest resolution
        return components

    # -------------------------------------------------
    # 4. STATISTICAL FEATURES
    # -------------------------------------------------
    def extract_statistical_features(self, components):
        """
        Extract mean, variance, skewness, kurtosis
        """
        features = []

        for comp in components:
            comp_flat = comp.flatten()

            mean = np.mean(comp_flat)
            var = np.var(comp_flat)
            skew = self._skewness(comp_flat)
            kurt = self._kurtosis(comp_flat)

            features.extend([mean, var, skew, kurt])

        return np.array(features, dtype=np.float32)

    def _skewness(self, x):
        mean = np.mean(x)
        std = np.std(x) + 1e-8
        return np.mean(((x - mean) / std) ** 3)

    def _kurtosis(self, x):
        mean = np.mean(x)
        std = np.std(x) + 1e-8
        return np.mean(((x - mean) / std) ** 4)

    # -------------------------------------------------
    # FULL PIPELINE
    # -------------------------------------------------
    def process(self, image, image_tensor):
        """
        image: numpy array (H, W, C)
        image_tensor: torch tensor (1, C, H, W)

        Returns:
        - feature vector
        - saliency map
        """

        # Step 1: Saliency
        saliency = self.compute_saliency(image_tensor)

        # Step 2: Adaptive filtering
        filtered = self.apply_saliency_filter(image, saliency)

        # Step 3: Multi-resolution decomposition
        components = self.multi_resolution_decomposition(filtered)

        # Step 4: Feature extraction
        features = self.extract_statistical_features(components)

        return features, saliency