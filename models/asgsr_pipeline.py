```python
import numpy as np
import cv2
import torch


class ASGSRPipeline:
    """
    Adaptive Saliency-Guided Signal Representation (ASGSR)

    Pipeline:
    1. Predicted-class saliency estimation
    2. Adaptive spatial filtering
    3. Multi-resolution signal decomposition
    4. Statistical feature extraction

    The ground-truth class is NOT required during inference.
    Saliency is computed with respect to the class predicted
    by the trained saliency-estimation CNN.
    """

    def __init__(self, model, device="cpu", num_levels=3):
        self.model = model
        self.device = device
        self.num_levels = num_levels

        self.model.to(self.device)
        self.model.eval()

    # -------------------------------------------------
    # 1. PREDICTED-CLASS SALIENCY ESTIMATION
    # -------------------------------------------------
    def compute_saliency(self, image_tensor):
        """
        Compute gradient-based saliency with respect to
        the predicted class.

        Ground-truth labels are NOT used.
        """

        image_tensor = (
            image_tensor.clone()
            .detach()
            .to(self.device)
        )

        image_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(image_tensor)

        # Predicted class
        predicted_class = torch.argmax(
            output,
            dim=1
        )

        # Score of predicted class
        score = output[
            0,
            predicted_class[0]
        ]

        # Clear model gradients
        self.model.zero_grad()

        # Gradient of predicted-class score
        score.backward()

        # Absolute input gradient
        saliency = (
            image_tensor.grad
            .detach()
            .abs()
        )

        # Aggregate channels
        saliency, _ = torch.max(
            saliency,
            dim=1
        )

        # Convert to NumPy
        saliency = (
            saliency
            .squeeze()
            .cpu()
            .numpy()
        )

        # Normalize saliency to [0, 1]
        saliency_min = saliency.min()
        saliency_max = saliency.max()

        if saliency_max > saliency_min:
            saliency = (
                saliency - saliency_min
            ) / (
                saliency_max - saliency_min
            )
        else:
            saliency = np.zeros_like(
                saliency,
                dtype=np.float32
            )

        return saliency.astype(np.float32)

    # -------------------------------------------------
    # 2. ADAPTIVE SALIENCY-GUIDED FILTERING
    # -------------------------------------------------
    def apply_saliency_filter(
        self,
        image,
        saliency
    ):
        """
        Apply normalized saliency as a spatial
        weighting function.

        Since saliency is normalized to [0,1],
        the filtering operation is energy-bounded.
        """

        image = np.asarray(
            image,
            dtype=np.float32
        )

        saliency = np.asarray(
            saliency,
            dtype=np.float32
        )

        # Expand saliency for multi-channel images
        if image.ndim == 3:
            saliency = np.expand_dims(
                saliency,
                axis=-1
            )

        # Spatially weighted representation
        filtered = image * saliency

        return filtered.astype(
            np.float32
        )

    # -------------------------------------------------
    # 3. MULTI-RESOLUTION DECOMPOSITION
    # -------------------------------------------------
    def multi_resolution_decomposition(
        self,
        image
    ):
        """
        Laplacian-pyramid-based
        multi-resolution decomposition.
        """

        components = []

        current = image.copy()

        for _ in range(self.num_levels):

            down = cv2.pyrDown(
                current
            )

            up = cv2.pyrUp(
                down,
                dstsize=(
                    current.shape[1],
                    current.shape[0]
                )
            )

            # High-frequency residual
            laplacian = (
                current - up
            )

            components.append(
                laplacian
            )

            current = down

        # Lowest-resolution component
        components.append(
            current
        )

        return components

    # -------------------------------------------------
    # 4. STATISTICAL FEATURE EXTRACTION
    # -------------------------------------------------
    def extract_statistical_features(
        self,
        components
    ):
        """
        Extract:
        - Mean
        - Variance
        - Skewness
        - Kurtosis
        """

        features = []

        for comp in components:

            comp_flat = (
                comp
                .astype(np.float32)
                .flatten()
            )

            mean = np.mean(
                comp_flat
            )

            variance = np.var(
                comp_flat
            )

            skewness = (
                self._skewness(
                    comp_flat
                )
            )

            kurtosis = (
                self._kurtosis(
                    comp_flat
                )
            )

            features.extend([
                mean,
                variance,
                skewness,
                kurtosis
            ])

        return np.asarray(
            features,
            dtype=np.float32
        )

    # -------------------------------------------------
    # SKEWNESS
    # -------------------------------------------------
    def _skewness(self, x):

        mean = np.mean(x)
        std = np.std(x) + 1e-8

        return np.mean(
            ((x - mean) / std) ** 3
        )

    # -------------------------------------------------
    # KURTOSIS
    # -------------------------------------------------
    def _kurtosis(self, x):

        mean = np.mean(x)
        std = np.std(x) + 1e-8

        return np.mean(
            ((x - mean) / std) ** 4
        )

    # -------------------------------------------------
    # COMPLETE ASGSR PROCESS
    # -------------------------------------------------
    def process(
        self,
        image,
        image_tensor
    ):
        """
        Returns:

        features
            Statistical ASGSR feature vector.

        saliency
            Predicted-class saliency map.

        filtered
            Saliency-guided spatial representation.
        """

        # ---------------------------------------------
        # Step 1: Predicted-class saliency
        # ---------------------------------------------
        saliency = (
            self.compute_saliency(
                image_tensor
            )
        )

        # ---------------------------------------------
        # Step 2: Adaptive filtering
        # ---------------------------------------------
        filtered = (
            self.apply_saliency_filter(
                image,
                saliency
            )
        )

        # ---------------------------------------------
        # Step 3: Multi-resolution decomposition
        # ---------------------------------------------
        components = (
            self.multi_resolution_decomposition(
                filtered
            )
        )

        # ---------------------------------------------
        # Step 4: Statistical features
        # ---------------------------------------------
        features = (
            self.extract_statistical_features(
                components
            )
        )

        return (
            features,
            saliency,
            filtered
        )
```
