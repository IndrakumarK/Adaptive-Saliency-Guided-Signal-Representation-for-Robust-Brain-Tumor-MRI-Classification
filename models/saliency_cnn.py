import numpy as np
import cv2
import torch


class ASGSRPipeline:
    """
    Adaptive Saliency-Guided Signal Representation (ASGSR)

    Pipeline:
    1. Gradient-based saliency estimation
    2. Saliency-guided adaptive filtering
    3. Multi-resolution signal decomposition
    4. Statistical feature extraction
    """

    def __init__(self, model, device="cpu", num_levels=3):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Trained lightweight CNN used for gradient-based
            signal sensitivity estimation.

        device : str
            Computational device ("cpu" or "cuda").

        num_levels : int
            Number of multi-resolution decomposition levels.
        """

        self.model = model
        self.device = device
        self.num_levels = num_levels

        # Move trained CNN to the selected device
        self.model.to(self.device)

        # Evaluation mode for inference/saliency generation
        self.model.eval()

    # =================================================
    # 1. SALIENCY ESTIMATION
    # =================================================
    def compute_saliency(self, image_tensor):
        """
        Compute a gradient-based saliency map with respect
        to the predicted class.

        Ground-truth class information is NOT required
        during inference.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Input MRI tensor of shape (1, C, H, W).

        Returns
        -------
        saliency : numpy.ndarray
            Normalized saliency map in the range [0, 1].
        """

        # Create an independent tensor for gradient computation
        image_tensor = (
            image_tensor
            .clone()
            .detach()
            .to(self.device)
        )

        # Enable gradient computation with respect to input
        image_tensor.requires_grad_(True)

        # ---------------------------------------------
        # Forward pass through trained saliency CNN
        # ---------------------------------------------
        output = self.model(image_tensor)

        # ---------------------------------------------
        # Determine predicted class
        # ---------------------------------------------
        predicted_class = torch.argmax(
            output,
            dim=1
        )

        # ---------------------------------------------
        # Select predicted-class score
        # ---------------------------------------------
        score = output[
            0,
            predicted_class[0]
        ]

        # ---------------------------------------------
        # Clear previously accumulated model gradients
        # ---------------------------------------------
        self.model.zero_grad(set_to_none=True)

        # ---------------------------------------------
        # Gradient of predicted-class score
        # with respect to input MRI
        # ---------------------------------------------
        score.backward()

        # ---------------------------------------------
        # Absolute gradient represents
        # gradient-based signal sensitivity
        # ---------------------------------------------
        saliency = (
            image_tensor.grad
            .detach()
            .abs()
        )

        # ---------------------------------------------
        # Aggregate gradients across channels
        # ---------------------------------------------
        saliency, _ = torch.max(
            saliency,
            dim=1
        )

        # ---------------------------------------------
        # Convert to NumPy
        # ---------------------------------------------
        saliency = (
            saliency
            .squeeze()
            .cpu()
            .numpy()
        )

        # ---------------------------------------------
        # Normalize saliency to [0, 1]
        # ---------------------------------------------
        saliency_min = saliency.min()
        saliency_max = saliency.max()

        if saliency_max > saliency_min:

            saliency = (
                saliency - saliency_min
            ) / (
                saliency_max - saliency_min
            )

        else:

            # Handle constant-gradient case
            saliency = np.zeros_like(
                saliency,
                dtype=np.float32
            )

        return saliency.astype(np.float32)

    # =================================================
    # 2. SALIENCY-GUIDED ADAPTIVE FILTERING
    # =================================================
    def apply_saliency_filter(self, image, saliency):
        """
        Apply saliency-guided spatial weighting
        to the MRI signal.

        Parameters
        ----------
        image : numpy.ndarray
            MRI image of shape (H, W, C).

        saliency : numpy.ndarray
            Saliency map of shape (H, W).

        Returns
        -------
        filtered : numpy.ndarray
            Saliency-weighted MRI signal.
        """

        # ---------------------------------------------
        # Match saliency dimensions with image channels
        # ---------------------------------------------
        if image.ndim == 3:

            saliency = np.expand_dims(
                saliency,
                axis=-1
            )

        # ---------------------------------------------
        # Element-wise spatial weighting
        # ---------------------------------------------
        filtered = image * saliency

        return filtered.astype(
            np.float32
        )

    # =================================================
    # 3. MULTI-RESOLUTION SIGNAL DECOMPOSITION
    # =================================================
    def multi_resolution_decomposition(self, image):
        """
        Perform multi-resolution signal decomposition
        using a Laplacian pyramid.

        Parameters
        ----------
        image : numpy.ndarray
            Saliency-filtered MRI signal.

        Returns
        -------
        components : list
            Multi-resolution signal components.
        """

        components = []

        # Initial signal
        current = image.copy()

        # ---------------------------------------------
        # Construct Laplacian pyramid
        # ---------------------------------------------
        for _ in range(self.num_levels):

            # Downsample
            down = cv2.pyrDown(
                current
            )

            # Upsample to original current resolution
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

            # Continue at lower resolution
            current = down

        # ---------------------------------------------
        # Add lowest-resolution component
        # ---------------------------------------------
        components.append(
            current
        )

        return components

    # =================================================
    # 4. STATISTICAL FEATURE EXTRACTION
    # =================================================
    def extract_statistical_features(self, components):
        """
        Extract statistical descriptors from each
        multi-resolution signal component.

        Descriptors:
        - Mean
        - Variance
        - Skewness
        - Kurtosis

        Parameters
        ----------
        components : list
            Multi-resolution signal components.

        Returns
        -------
        features : numpy.ndarray
            Statistical feature vector.
        """

        features = []

        for comp in components:

            # -----------------------------------------
            # Flatten signal component
            # -----------------------------------------
            comp_flat = comp.flatten()

            # -----------------------------------------
            # Mean
            # -----------------------------------------
            mean = np.mean(
                comp_flat
            )

            # -----------------------------------------
            # Variance
            # -----------------------------------------
            var = np.var(
                comp_flat
            )

            # -----------------------------------------
            # Skewness
            # -----------------------------------------
            skew = self._skewness(
                comp_flat
            )

            # -----------------------------------------
            # Kurtosis
            # -----------------------------------------
            kurt = self._kurtosis(
                comp_flat
            )

            features.extend([
                mean,
                var,
                skew,
                kurt
            ])

        return np.asarray(
            features,
            dtype=np.float32
        )

    # =================================================
    # 5. SKEWNESS
    # =================================================
    def _skewness(self, x):
        """
        Compute standardized third central moment.
        """

        mean = np.mean(x)

        std = (
            np.std(x) + 1e-8
        )

        return np.mean(
            ((x - mean) / std) ** 3
        )

    # =================================================
    # 6. KURTOSIS
    # =================================================
    def _kurtosis(self, x):
        """
        Compute standardized fourth central moment.
        """

        mean = np.mean(x)

        std = (
            np.std(x) + 1e-8
        )

        return np.mean(
            ((x - mean) / std) ** 4
        )

    # =================================================
    # 7. COMPLETE ASGSR PIPELINE
    # =================================================
    def process(self, image, image_tensor):
        """
        Execute the complete ASGSR processing pipeline.

        Parameters
        ----------
        image : numpy.ndarray
            MRI image of shape (H, W, C).

        image_tensor : torch.Tensor
            MRI tensor of shape (1, C, H, W).

        Returns
        -------
        features : numpy.ndarray
            Extracted statistical feature vector.

        saliency : numpy.ndarray
            Gradient-based saliency map.
        """

        # ---------------------------------------------
        # Step 1: Gradient-based saliency estimation
        # using the predicted class
        # ---------------------------------------------
        saliency = self.compute_saliency(
            image_tensor
        )

        # ---------------------------------------------
        # Step 2: Saliency-guided adaptive filtering
        # ---------------------------------------------
        filtered = self.apply_saliency_filter(
            image,
            saliency
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
        # Step 4: Statistical feature extraction
        # ---------------------------------------------
        features = (
            self.extract_statistical_features(
                components
            )
        )

        return features, saliency
