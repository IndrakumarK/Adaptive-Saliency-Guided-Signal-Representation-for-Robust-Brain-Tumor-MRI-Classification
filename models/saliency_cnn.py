"""
ASGSR MRI Classification
Adaptive Saliency-Guided Signal Representation

Pipeline:
    1. Predicted-class gradient-based saliency estimation
    2. Saliency-guided spatial weighting
    3. Multi-resolution signal decomposition
    4. Statistical feature extraction

The CNN is used as a signal-sensitivity/saliency probe.
The extracted statistical representation can subsequently be
provided to the Bayesian classifier.
"""

import numpy as np
import cv2
import torch


class ASGSRPipeline:
    """
    Adaptive Saliency-Guided Signal Representation (ASGSR).

    The pipeline converts an MRI image into a compact
    saliency-guided statistical signal representation.

    Processing stages
    -----------------
    1. Predicted-class gradient saliency
    2. Saliency-guided spatial weighting
    3. Multi-resolution Laplacian decomposition
    4. Statistical feature extraction

    Parameters
    ----------
    model : torch.nn.Module
        Trained CNN used as the gradient-based signal
        sensitivity estimator.

    device : str, optional
        Computation device, by default "cpu".

    num_levels : int, optional
        Number of Laplacian decomposition levels.
        Default is 4.
    """

    def __init__(
        self,
        model,
        device="cpu",
        num_levels=4
    ):
        self.model = model
        self.device = device
        self.num_levels = int(num_levels)

        if self.num_levels < 1:
            raise ValueError(
                "num_levels must be greater than or equal to 1."
            )

        # Move the trained saliency CNN to the selected device.
        self.model.to(self.device)

        # Saliency generation is performed in evaluation mode.
        self.model.eval()

    # =========================================================
    # 1. SALIENCY ESTIMATION
    # =========================================================

    def compute_saliency(self, image_tensor):
        """
        Compute a predicted-class gradient-based saliency map.

        The predicted class is obtained directly from the CNN
        output. Ground-truth labels are therefore not required
        during saliency generation.

        Parameters
        ----------
        image_tensor : torch.Tensor
            MRI tensor with shape:

                (1, C, H, W)

        Returns
        -------
        np.ndarray
            Normalized saliency map with shape (H, W)
            and values in [0, 1].
        """

        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError(
                "image_tensor must be a torch.Tensor."
            )

        if image_tensor.ndim != 4:
            raise ValueError(
                "image_tensor must have shape (1, C, H, W)."
            )

        # -----------------------------------------------------
        # Create an independent tensor for gradient computation
        # -----------------------------------------------------

        x = (
            image_tensor
            .clone()
            .detach()
            .to(self.device)
        )

        x.requires_grad_(True)

        # -----------------------------------------------------
        # Forward pass through the trained CNN
        # -----------------------------------------------------

        self.model.zero_grad(set_to_none=True)

        output = self.model(x)

        if output.ndim != 2:
            raise ValueError(
                "The saliency CNN must return logits with shape "
                "(batch_size, num_classes)."
            )

        # -----------------------------------------------------
        # Determine the predicted class
        # -----------------------------------------------------

        predicted_class = torch.argmax(
            output,
            dim=1
        )

        # -----------------------------------------------------
        # Select predicted-class response
        # -----------------------------------------------------

        score = output[
            0,
            predicted_class[0]
        ]

        # -----------------------------------------------------
        # Compute gradient with respect to the input MRI
        # -----------------------------------------------------

        score.backward()

        if x.grad is None:
            raise RuntimeError(
                "Input gradient could not be computed. "
                "Ensure that the CNN output is differentiable "
                "with respect to the input."
            )

        # -----------------------------------------------------
        # Absolute input gradient = signal sensitivity
        # -----------------------------------------------------

        saliency = (
            x.grad
            .detach()
            .abs()
        )

        # -----------------------------------------------------
        # Aggregate across MRI channels/modalities
        # -----------------------------------------------------

        saliency, _ = torch.max(
            saliency,
            dim=1
        )

        # -----------------------------------------------------
        # Convert to NumPy
        # -----------------------------------------------------

        saliency = (
            saliency
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # -----------------------------------------------------
        # Max normalization
        #
        # s_norm = s / max(s)
        # -----------------------------------------------------

        saliency_max = float(
            np.max(saliency)
        )

        if saliency_max > 1e-8:

            saliency = (
                saliency /
                saliency_max
            )

        else:

            saliency = np.zeros_like(
                saliency,
                dtype=np.float32
            )

        # Numerical safety.
        saliency = np.clip(
            saliency,
            0.0,
            1.0
        )

        return saliency.astype(
            np.float32
        )

    # =========================================================
    # 2. SALIENCY-GUIDED SPATIAL WEIGHTING
    # =========================================================

    def apply_saliency_filter(
        self,
        image,
        saliency
    ):
        """
        Apply the saliency map as a spatial weighting function.

        Each spatial location of the MRI signal is weighted
        according to its predicted-class sensitivity.

        Parameters
        ----------
        image : np.ndarray
            MRI image with shape (H, W) or (H, W, C).

        saliency : np.ndarray
            Saliency map with shape (H, W).

        Returns
        -------
        np.ndarray
            Saliency-weighted MRI signal.
        """

        if not isinstance(
            image,
            np.ndarray
        ):
            image = np.asarray(
                image,
                dtype=np.float32
            )

        if not isinstance(
            saliency,
            np.ndarray
        ):
            saliency = np.asarray(
                saliency,
                dtype=np.float32
            )

        if saliency.ndim != 2:
            raise ValueError(
                "Saliency map must have shape (H, W)."
            )

        if image.ndim not in (2, 3):
            raise ValueError(
                "Image must have shape (H, W) or (H, W, C)."
            )

        # -----------------------------------------------------
        # Check spatial compatibility
        # -----------------------------------------------------

        if (
            image.shape[0] != saliency.shape[0]
            or
            image.shape[1] != saliency.shape[1]
        ):
            raise ValueError(
                "Image and saliency map must have identical "
                "spatial dimensions."
            )

        # -----------------------------------------------------
        # Ensure saliency is in [0, 1]
        # -----------------------------------------------------

        saliency = np.clip(
            saliency,
            0.0,
            1.0
        ).astype(
            np.float32
        )

        # -----------------------------------------------------
        # Expand saliency map for multi-channel MRI
        # -----------------------------------------------------

        if image.ndim == 3:

            saliency_weight = np.expand_dims(
                saliency,
                axis=-1
            )

        else:

            saliency_weight = saliency

        # -----------------------------------------------------
        # Element-wise spatial weighting
        # -----------------------------------------------------

        filtered = (
            image.astype(np.float32)
            *
            saliency_weight
        )

        return filtered.astype(
            np.float32
        )

    # =========================================================
    # 3. MULTI-RESOLUTION SIGNAL DECOMPOSITION
    # =========================================================

    def multi_resolution_decomposition(
        self,
        image
    ):
        """
        Perform multi-resolution signal decomposition
        using a Laplacian pyramid.

        Each level contains a high-frequency residual,
        followed by a final low-resolution component.

        Parameters
        ----------
        image : np.ndarray
            Saliency-weighted MRI signal.

        Returns
        -------
        list of np.ndarray
            Multi-resolution signal components.
        """

        if not isinstance(
            image,
            np.ndarray
        ):
            image = np.asarray(
                image,
                dtype=np.float32
            )

        if image.ndim not in (2, 3):
            raise ValueError(
                "Input image must have shape "
                "(H, W) or (H, W, C)."
            )

        components = []

        # -----------------------------------------------------
        # Initial signal
        # -----------------------------------------------------

        current = image.astype(
            np.float32,
            copy=True
        )

        # -----------------------------------------------------
        # Construct Laplacian pyramid
        # -----------------------------------------------------

        for level in range(
            self.num_levels
        ):

            # Safety check for very small images.
            if (
                current.shape[0] < 4
                or
                current.shape[1] < 4
            ):
                break

            # -------------------------------------------------
            # Downsample
            # -------------------------------------------------

            down = cv2.pyrDown(
                current
            )

            # -------------------------------------------------
            # Upsample to current resolution
            # -------------------------------------------------

            up = cv2.pyrUp(
                down,
                dstsize=(
                    current.shape[1],
                    current.shape[0]
                )
            )

            # -------------------------------------------------
            # High-frequency residual
            # -------------------------------------------------

            laplacian = (
                current - up
            )

            components.append(
                laplacian.astype(
                    np.float32
                )
            )

            # -------------------------------------------------
            # Continue at lower resolution
            # -------------------------------------------------

            current = down.astype(
                np.float32
            )

        # -----------------------------------------------------
        # Final low-resolution component
        # -----------------------------------------------------

        components.append(
            current.astype(
                np.float32
            )
        )

        return components

    # =========================================================
    # 4. STATISTICAL FEATURE EXTRACTION
    # =========================================================

    def extract_statistical_features(
        self,
        components
    ):
        """
        Extract statistical descriptors from each
        multi-resolution component.

        Descriptors
        -----------
        1. Mean
        2. Variance
        3. Skewness
        4. Kurtosis

        For L decomposition levels, the resulting feature
        vector contains:

            4 × (L + 1)

        features.

        Parameters
        ----------
        components : list of np.ndarray
            Multi-resolution signal components.

        Returns
        -------
        np.ndarray
            Statistical feature vector.
        """

        if not isinstance(
            components,
            (list, tuple)
        ):
            raise TypeError(
                "components must be a list or tuple "
                "of NumPy arrays."
            )

        if len(components) == 0:
            raise ValueError(
                "No multi-resolution components were provided."
            )

        features = []

        for component in components:

            if component is None:
                raise ValueError(
                    "A decomposition component is None."
                )

            comp = np.asarray(
                component,
                dtype=np.float64
            )

            # -------------------------------------------------
            # Flatten component
            # -------------------------------------------------

            values = comp.reshape(
                -1
            )

            # -------------------------------------------------
            # Remove non-finite values
            # -------------------------------------------------

            values = values[
                np.isfinite(values)
            ]

            if values.size == 0:

                mean = 0.0
                variance = 0.0
                skewness = 0.0
                kurtosis = 0.0

            else:

                # ---------------------------------------------
                # Mean
                # ---------------------------------------------

                mean = float(
                    np.mean(values)
                )

                # ---------------------------------------------
                # Variance
                # ---------------------------------------------

                variance = float(
                    np.var(values)
                )

                # ---------------------------------------------
                # Skewness
                # ---------------------------------------------

                skewness = float(
                    self._skewness(values)
                )

                # ---------------------------------------------
                # Kurtosis
                # ---------------------------------------------

                kurtosis = float(
                    self._kurtosis(values)
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

    # =========================================================
    # 5. SKEWNESS
    # =========================================================

    @staticmethod
    def _skewness(x):
        """
        Compute the standardized third central moment.

        Skewness:
            E[((x - μ) / σ)^3]
        """

        x = np.asarray(
            x,
            dtype=np.float64
        )

        if x.size == 0:
            return 0.0

        mean = np.mean(x)

        std = np.std(x)

        if std < 1e-12:
            return 0.0

        value = np.mean(
            ((x - mean) / std) ** 3
        )

        if not np.isfinite(value):
            return 0.0

        return float(value)

    # =========================================================
    # 6. KURTOSIS
    # =========================================================

    @staticmethod
    def _kurtosis(x):
        """
        Compute the standardized fourth central moment.

        Note:
            This is the fourth standardized moment itself,
            not excess kurtosis.

        Kurtosis:
            E[((x - μ) / σ)^4]
        """

        x = np.asarray(
            x,
            dtype=np.float64
        )

        if x.size == 0:
            return 0.0

        mean = np.mean(x)

        std = np.std(x)

        if std < 1e-12:
            return 0.0

        value = np.mean(
            ((x - mean) / std) ** 4
        )

        if not np.isfinite(value):
            return 0.0

        return float(value)

    # =========================================================
    # 7. COMPLETE ASGSR PIPELINE
    # =========================================================

    def process(
        self,
        image,
        image_tensor
    ):
        """
        Execute the complete ASGSR processing pipeline.

        Pipeline:
            MRI
              ↓
            Predicted-class saliency
              ↓
            Saliency-guided spatial weighting
              ↓
            Multi-resolution decomposition
              ↓
            Statistical descriptors
              ↓
            Feature vector

        Parameters
        ----------
        image : np.ndarray
            MRI image with shape (H, W) or (H, W, C).

        image_tensor : torch.Tensor
            Corresponding CNN input tensor with shape
            (1, C, H, W).

        Returns
        -------
        features : np.ndarray
            Statistical ASGSR feature vector.

        saliency : np.ndarray
            Predicted-class gradient saliency map.
        """

        # -----------------------------------------------------
        # Step 1
        # Predicted-class gradient saliency
        # -----------------------------------------------------

        saliency = self.compute_saliency(
            image_tensor
        )

        # -----------------------------------------------------
        # Step 2
        # Saliency-guided spatial weighting
        # -----------------------------------------------------

        filtered = self.apply_saliency_filter(
            image,
            saliency
        )

        # -----------------------------------------------------
        # Step 3
        # Multi-resolution decomposition
        # -----------------------------------------------------

        components = (
            self.multi_resolution_decomposition(
                filtered
            )
        )

        # -----------------------------------------------------
        # Step 4
        # Statistical feature extraction
        # -----------------------------------------------------

        features = (
            self.extract_statistical_features(
                components
            )
        )

        return features, saliency

    # =========================================================
    # 8. OPTIONAL COMPONENT-LEVEL PROCESSING
    # =========================================================

    def process_with_components(
        self,
        image,
        image_tensor
    ):
        """
        Execute ASGSR while additionally returning the
        intermediate decomposition components.

        Useful for debugging, visualization, and
        reproducibility analysis.

        Returns
        -------
        dict
            Dictionary containing:
                features
                saliency
                filtered_image
                components
        """

        # Saliency
        saliency = self.compute_saliency(
            image_tensor
        )

        # Spatial weighting
        filtered = self.apply_saliency_filter(
            image,
            saliency
        )

        # Multi-resolution decomposition
        components = (
            self.multi_resolution_decomposition(
                filtered
            )
        )

        # Statistical representation
        features = (
            self.extract_statistical_features(
                components
            )
        )

        return {
            "features": features,
            "saliency": saliency,
            "filtered_image": filtered,
            "components": components
        }


# =============================================================
# SIMPLE SELF-TEST
# =============================================================

if __name__ == "__main__":

    class DummyCNN(torch.nn.Module):

        def __init__(self):
            super().__init__()

            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(
                    4,
                    32,
                    kernel_size=3,
                    padding=1
                ),
                torch.nn.ReLU(),

                torch.nn.Conv2d(
                    32,
                    64,
                    kernel_size=3,
                    padding=1
                ),
                torch.nn.ReLU(),

                torch.nn.AdaptiveAvgPool2d(
                    (1, 1)
                )
            )

            self.classifier = torch.nn.Linear(
                64,
                4
            )

        def forward(self, x):

            x = self.features(x)

            x = x.flatten(
                1
            )

            return self.classifier(x)

    # ---------------------------------------------------------
    # Test configuration
    # ---------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = DummyCNN()

    pipeline = ASGSRPipeline(
        model=model,
        device=device,
        num_levels=4
    )

    # Four-channel MRI example.
    image_tensor = torch.randn(
        1,
        4,
        224,
        224
    )

    image = (
        image_tensor[0]
        .permute(1, 2, 0)
        .numpy()
        .astype(np.float32)
    )

    # ---------------------------------------------------------
    # Execute pipeline
    # ---------------------------------------------------------

    features, saliency = pipeline.process(
        image,
        image_tensor
    )

    print(
        "Saliency shape:",
        saliency.shape
    )

    print(
        "Saliency range:",
        float(saliency.min()),
        "to",
        float(saliency.max())
    )

    print(
        "Feature vector shape:",
        features.shape
    )

    print(
        "Number of extracted features:",
        len(features)
    )
