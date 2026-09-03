import numpy as np
import cv2
import torch


class ASGSRPipeline:
    """
    Adaptive Saliency-Guided Signal Representation (ASGSR)

    Pipeline:
    1. Predicted-class saliency estimation
    2. Adaptive saliency-guided spatial filtering
    3. Multi-resolution signal decomposition
    4. Statistical feature extraction

    During inference, the ground-truth class is not required.
    The saliency map is computed with respect to the class
    predicted by the trained saliency-estimation CNN.
    """

    def __init__(self, model, device="cpu", num_levels=4):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Trained saliency-estimation CNN.

        device : str or torch.device
            Computation device.

        num_levels : int
            Number of multi-resolution decomposition levels.
            The proposed implementation uses K = 4.
        """

        self.model = model
        self.device = device
        self.num_levels = num_levels

        self.model.to(self.device)
        self.model.eval()

    # =================================================
    # 1. PREDICTED-CLASS SALIENCY ESTIMATION
    # =================================================
    def compute_saliency(self, image_tensor):
        """
        Compute gradient-based saliency with respect to
        the predicted class.

        Ground-truth labels are NOT used during inference.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Input tensor of shape (1, C, H, W).

        Returns
        -------
        saliency : numpy.ndarray
            Normalized saliency map in the range [0, 1].
        """

        # ---------------------------------------------
        # Create independent tensor for gradient
        # computation
        # ---------------------------------------------
        image_tensor = (
            image_tensor
            .clone()
            .detach()
            .to(self.device)
        )

        image_tensor.requires_grad_(True)

        # ---------------------------------------------
        # Forward pass through trained CNN
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
        # Clear previously accumulated gradients
        # ---------------------------------------------
        self.model.zero_grad()

        # ---------------------------------------------
        # Compute gradient of predicted-class score
        # with respect to the input MRI
        # ---------------------------------------------
        score.backward()

        # ---------------------------------------------
        # Absolute gradient represents sensitivity
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

            saliency = np.zeros_like(
                saliency,
                dtype=np.float32
            )

        return saliency.astype(
            np.float32
        )

    # =================================================
    # 2. ADAPTIVE SALIENCY-GUIDED FILTERING
    # =================================================
    def apply_saliency_filter(
        self,
        image,
        saliency
    ):
        """
        Apply normalized saliency as a spatial
        weighting function.

        Since the saliency map is normalized to [0, 1],
        the resulting spatial weighting does not
        amplify the magnitude of the input signal.

        Parameters
        ----------
        image : numpy.ndarray
            Input MRI image of shape (H, W, C).

        saliency : numpy.ndarray
            Saliency map of shape (H, W).

        Returns
        -------
        filtered : numpy.ndarray
            Saliency-weighted MRI representation.
        """

        # ---------------------------------------------
        # Convert input to float32
        # ---------------------------------------------
        image = np.asarray(
            image,
            dtype=np.float32
        )

        saliency = np.asarray(
            saliency,
            dtype=np.float32
        )

        # ---------------------------------------------
        # Expand saliency map for multi-channel image
        # ---------------------------------------------
        if image.ndim == 3:

            saliency = np.expand_dims(
                saliency,
                axis=-1
            )

        # ---------------------------------------------
        # Element-wise spatial weighting
        # ---------------------------------------------
        filtered = (
            image * saliency
        )

        return filtered.astype(
            np.float32
        )

    # =================================================
    # 3. MULTI-RESOLUTION DECOMPOSITION
    # =================================================
    def multi_resolution_decomposition(
        self,
        image
    ):
        """
        Perform multi-resolution signal decomposition
        using a Laplacian pyramid.

        The proposed implementation uses K = 4
        decomposition levels.

        Parameters
        ----------
        image : numpy.ndarray
            Saliency-filtered MRI representation.

        Returns
        -------
        components : list
            Multi-resolution signal components.
        """

        components = []

        # ---------------------------------------------
        # Initial signal
        # ---------------------------------------------
        current = image.copy()

        # ---------------------------------------------
        # Hierarchical decomposition
        # ---------------------------------------------
        for level in range(
            self.num_levels
        ):

            # -----------------------------------------
            # Downsample
            # -----------------------------------------
            down = cv2.pyrDown(
                current
            )

            # -----------------------------------------
            # Upsample to current resolution
            # -----------------------------------------
            up = cv2.pyrUp(
                down,
                dstsize=(
                    current.shape[1],
                    current.shape[0]
                )
            )

            # -----------------------------------------
            # High-frequency residual
            # -----------------------------------------
            laplacian = (
                current - up
            )

            components.append(
                laplacian
            )

            # -----------------------------------------
            # Continue at lower resolution
            # -----------------------------------------
            current = down

        # ---------------------------------------------
        # Lowest-resolution component
        # ---------------------------------------------
        components.append(
            current
        )

        return components

    # =================================================
    # 4. STATISTICAL FEATURE EXTRACTION
    # =================================================
    def extract_statistical_features(
        self,
        components
    ):
        """
        Extract statistical descriptors from each
        multi-resolution component.

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
            # Flatten component
            # -----------------------------------------
            comp_flat = (
                comp
                .astype(np.float32)
                .flatten()
            )

            # -----------------------------------------
            # Mean
            # -----------------------------------------
            mean = np.mean(
                comp_flat
            )

            # -----------------------------------------
            # Variance
            # -----------------------------------------
            variance = np.var(
                comp_flat
            )

            # -----------------------------------------
            # Skewness
            # -----------------------------------------
            skewness = (
                self._skewness(
                    comp_flat
                )
            )

            # -----------------------------------------
            # Kurtosis
            # -----------------------------------------
            kurtosis = (
                self._kurtosis(
                    comp_flat
                )
            )

            # -----------------------------------------
            # Store descriptors
            # -----------------------------------------
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

    # =================================================
    # SKEWNESS
    # =================================================
    def _skewness(self, x):
        """
        Compute standardized third central moment.
        """

        mean = np.mean(x)

        std = (
            np.std(x)
            + 1e-8
        )

        return np.mean(
            ((x - mean) / std) ** 3
        )

    # =================================================
    # KURTOSIS
    # =================================================
    def _kurtosis(self, x):
        """
        Compute standardized fourth central moment.
        """

        mean = np.mean(x)

        std = (
            np.std(x)
            + 1e-8
        )

        return np.mean(
            ((x - mean) / std) ** 4
        )

    # =================================================
    # COMPLETE ASGSR PROCESS
    # =================================================
    def process(
        self,
        image,
        image_tensor
    ):
        """
        Execute the complete ASGSR pipeline.

        Parameters
        ----------
        image : numpy.ndarray
            MRI image of shape (H, W, C).

        image_tensor : torch.Tensor
            MRI tensor of shape (1, C, H, W).

        Returns
        -------
        features : numpy.ndarray
            Statistical ASGSR feature vector.

        saliency : numpy.ndarray
            Predicted-class saliency map.

        filtered : numpy.ndarray
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
        # Step 2: Adaptive saliency-guided filtering
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
        # Step 4: Statistical feature extraction
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
