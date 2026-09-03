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
        model: Saliency CNN used for gradient-based
               signal sensitivity estimation
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
        Compute gradient-based saliency map with respect
        to the predicted class.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Input tensor of shape (1, C, H, W)

        Returns
        -------
        saliency : numpy.ndarray
            Normalized saliency map in the range [0, 1]
        """

        # Create an independent tensor for gradient computation
        image_tensor = image_tensor.clone().detach().to(self.device)
        image_tensor.requires_grad_(True)

        # Forward pass through the trained saliency CNN
        output = self.model(image_tensor)

        # Determine the predicted class
        predicted_class = torch.argmax(output, dim=1)

        # Select the score corresponding to the predicted class
        score = output[0, predicted_class[0]]

        # Clear previously accumulated gradients
        self.model.zero_grad()

        # Compute gradient of the predicted-class score
        # with respect to the input image
        score.backward()

        # Absolute gradient represents signal sensitivity
        saliency = image_tensor.grad.detach().abs()

        # Aggregate gradients across image channels
        saliency, _ = torch.max(saliency, dim=1)

        # Convert to NumPy
        saliency = saliency.squeeze().cpu().numpy()

        # Normalize saliency map to [0, 1]
        saliency_min = saliency.min()
        saliency_max = saliency.max()

        if saliency_max > saliency_min:
            saliency = (
                saliency - saliency_min
            ) / (saliency_max - saliency_min)
        else:
            saliency = np.zeros_like(saliency)

        return saliency

    # -------------------------------------------------
    # 2. ADAPTIVE FILTERING
    # -------------------------------------------------
    def apply_saliency_filter(self, image, saliency):
        """
        Apply saliency-guided spatial weighting to the MRI signal.

        Parameters
        ----------
        image : numpy.ndarray
            MRI image of shape (H, W, C)

        saliency : numpy.ndarray
            Saliency map of shape (H, W)

        Returns
        -------
        filtered : numpy.ndarray
            Saliency-weighted MRI signal
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
        Perform multi-resolution signal decomposition
        using a Laplacian pyramid.

        Parameters
        ----------
        image : numpy.ndarray
            Saliency-filtered MRI signal

        Returns
        -------
        components : list
            Multi-resolution signal components
        """

        components = []
        current = image.copy()

        for _ in range(self.num_levels):

            # Downsample
            down = cv2.pyrDown(current)

            # Upsample to the current resolution
            up = cv2.pyrUp(
                down,
                dstsize=(current.shape[1], current.shape[0])
            )

            # Compute high-frequency residual
            laplacian = current - up

            components.append(laplacian)

            # Continue decomposition at lower resolution
            current = down

        # Add lowest-resolution component
        components.append(current)

        return components

    # -------------------------------------------------
    # 4. STATISTICAL FEATURE EXTRACTION
    # -------------------------------------------------
    def extract_statistical_features(self, components):
        """
        Extract statistical descriptors from each
        multi-resolution signal component.

        Descriptors:
        - Mean
        - Variance
        - Skewness
        - Kurtosis
        """

        features = []

        for comp in components:

            # Flatten signal component
            comp_flat = comp.flatten()

            # Mean
            mean = np.mean(comp_flat)

            # Variance
            var = np.var(comp_flat)

            # Skewness
            skew = self._skewness(comp_flat)

            # Kurtosis
            kurt = self._kurtosis(comp_flat)

            features.extend([
                mean,
                var,
                skew,
                kurt
            ])

        return np.array(
            features,
            dtype=np.float32
        )

    # -------------------------------------------------
    # SKEWNESS
    # -------------------------------------------------
    def _skewness(self, x):
        """
        Compute standardized third central moment.
        """

        mean = np.mean(x)
        std = np.std(x) + 1e-8

        return np.mean(
            ((x - mean) / std) ** 3
        )

    # -------------------------------------------------
    # KURTOSIS
    # -------------------------------------------------
    def _kurtosis(self, x):
        """
        Compute standardized fourth central moment.
        """

        mean = np.mean(x)
        std = np.std(x) + 1e-8

        return np.mean(
            ((x - mean) / std) ** 4
        )

    # -------------------------------------------------
    # FULL ASGSR PIPELINE
    # -------------------------------------------------
    def process(self, image, image_tensor):
        """
        Execute the complete ASGSR processing pipeline.

        Parameters
        ----------
        image : numpy.ndarray
            MRI image of shape (H, W, C)

        image_tensor : torch.Tensor
            MRI tensor of shape (1, C, H, W)

        Returns
        -------
        features : numpy.ndarray
            Extracted statistical feature vector

        saliency : numpy.ndarray
            Gradient-based saliency map
        """

        # ---------------------------------------------
        # Step 1: Gradient-based saliency estimation
        # ---------------------------------------------
        saliency = self.compute_saliency(image_tensor)

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
        components = self.multi_resolution_decomposition(
            filtered
        )

        # ---------------------------------------------
        # Step 4: Statistical feature extraction
        # ---------------------------------------------
        features = self.extract_statistical_features(
            components
        )

        return features, saliency
