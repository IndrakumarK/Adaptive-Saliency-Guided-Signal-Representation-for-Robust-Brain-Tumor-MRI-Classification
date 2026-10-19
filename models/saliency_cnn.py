"""
ASGSR MRI Classification
Adaptive Saliency-Guided Signal Representation

Saliency CNN
------------

The CNN serves as the signal-sensitivity / saliency probe
within the ASGSR framework.

The network produces class logits that are used to obtain
predicted-class gradient-based saliency maps. The learned
representation is subsequently processed by the ASGSR
pipeline and classified using the Bayesian decision stage.

Training objectives implemented here:

    1. Classification loss
    2. Saliency smoothness regularization
    3. Mutual-information-inspired feature/class objective
    4. Confidence regularization

The implementation is intentionally lightweight so that the
CNN remains a saliency probe rather than replacing the
statistical + Bayesian classification framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================
# BASIC CONVOLUTIONAL BLOCK
# =============================================================

class ConvBlock(nn.Module):
    """
    Two-layer convolutional feature extraction block.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# =============================================================
# SALIENCY CNN
# =============================================================

class SaliencyCNN(nn.Module):
    """
    CNN used as the signal-sensitivity and saliency probe
    in the ASGSR framework.

    Parameters
    ----------
    num_classes : int
        Number of MRI classes.

    in_channels : int
        Number of MRI input channels/modalities.

    channels : tuple
        Number of channels in successive convolutional blocks.

    dropout : float
        Dropout probability used before classification.
    """

    def __init__(
        self,
        num_classes=4,
        in_channels=4,
        channels=(32, 64, 128, 256),
        dropout=0.20
    ):
        super().__init__()

        if len(channels) != 4:
            raise ValueError(
                "channels must contain four feature dimensions."
            )

        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)

        c1, c2, c3, c4 = channels

        # -----------------------------------------------------
        # Feature extraction
        # -----------------------------------------------------

        self.block1 = ConvBlock(
            in_channels,
            c1
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.block2 = ConvBlock(
            c1,
            c2
        )

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.block3 = ConvBlock(
            c2,
            c3
        )

        self.pool3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.block4 = ConvBlock(
            c3,
            c4
        )

        # -----------------------------------------------------
        # Global feature representation
        # -----------------------------------------------------

        self.global_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.dropout = nn.Dropout(
            p=float(dropout)
        )

        # -----------------------------------------------------
        # Classification head
        # -----------------------------------------------------

        self.classifier = nn.Linear(
            c4,
            self.num_classes
        )

        self.feature_dim = c4

        # Store feature maps for optional interpretability
        # and diagnostic analysis.
        self._last_feature_map = None
        self._last_feature_vector = None

    # =========================================================
    # FORWARD
    # =========================================================

    def forward(self, x):
        """
        Forward propagation.

        Parameters
        ----------
        x : torch.Tensor
            MRI tensor with shape:

                (B, C, H, W)

        Returns
        -------
        torch.Tensor
            Class logits with shape:

                (B, num_classes)
        """

        if not isinstance(x, torch.Tensor):
            raise TypeError(
                "Input must be a torch.Tensor."
            )

        if x.ndim != 4:
            raise ValueError(
                "Input must have shape (B, C, H, W)."
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, "
                f"but received {x.shape[1]}."
            )

        # Feature extraction
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = self.pool3(x)

        x = self.block4(x)

        # Preserve spatial feature map.
        self._last_feature_map = x

        # Global representation.
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        self._last_feature_vector = x

        x = self.dropout(x)

        logits = self.classifier(x)

        return logits

    # =========================================================
    # FEATURE EXTRACTION
    # =========================================================

    def extract_features(self, x):
        """
        Extract the compact CNN feature representation.

        This function does not perform Bayesian classification.
        It returns the feature representation that can be used
        for analysis or downstream processing.
        """

        if not isinstance(x, torch.Tensor):
            raise TypeError(
                "Input must be a torch.Tensor."
            )

        # Run the feature extractor explicitly so that the
        # representation can be obtained independently of the
        # classification head.

        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = self.pool3(x)

        x = self.block4(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        return x

    # =========================================================
    # PREDICTED-CLASS SCORE
    # =========================================================

    def predicted_class_score(self, x):
        """
        Return the predicted class and corresponding logit.

        The predicted-class response is the quantity used by
        the ASGSR pipeline for gradient-based saliency.
        """

        logits = self.forward(x)

        predicted_class = torch.argmax(
            logits,
            dim=1
        )

        score = logits[
            torch.arange(
                logits.shape[0],
                device=logits.device
            ),
            predicted_class
        ]

        return (
            logits,
            predicted_class,
            score
        )

    # =========================================================
    # PREDICTED-CLASS SALIENCY
    # =========================================================

    def compute_saliency(
        self,
        x,
        normalize=True
    ):
        """
        Compute predicted-class gradient saliency.

        Saliency is obtained from the absolute gradient of the
        predicted-class response with respect to the input MRI.

        Parameters
        ----------
        x : torch.Tensor
            Input MRI tensor (B, C, H, W).

        normalize : bool
            If True, normalize each sample by its maximum
            saliency value.

        Returns
        -------
        saliency : torch.Tensor
            Saliency map with shape (B, H, W).

        predicted_class : torch.Tensor
            Predicted class index for each sample.
        """

        if x.ndim != 4:
            raise ValueError(
                "Input must have shape (B, C, H, W)."
            )

        x_grad = x.clone().detach()
        x_grad.requires_grad_(True)

        self.zero_grad(set_to_none=True)

        logits = self.forward(
            x_grad
        )

        predicted_class = torch.argmax(
            logits,
            dim=1
        )

        batch_indices = torch.arange(
            logits.shape[0],
            device=logits.device
        )

        score = logits[
            batch_indices,
            predicted_class
        ].sum()

        score.backward()

        if x_grad.grad is None:
            raise RuntimeError(
                "Unable to obtain the input gradient."
            )

        # Absolute gradient.
        saliency = x_grad.grad.abs()

        # Aggregate across MRI modalities/channels.
        saliency = torch.max(
            saliency,
            dim=1
        ).values

        if normalize:

            maximum = (
                saliency
                .flatten(1)
                .amax(dim=1)
                .view(-1, 1, 1)
            )

            saliency = torch.where(
                maximum > 1e-8,
                saliency / maximum.clamp_min(1e-8),
                torch.zeros_like(saliency)
            )

        return (
            saliency.detach(),
            predicted_class.detach()
        )

    # =========================================================
    # SALIENCY SMOOTHNESS
    # =========================================================

    @staticmethod
    def saliency_smoothness_loss(
        saliency
    ):
        """
        Spatial smoothness regularization for the saliency map.

        The loss penalizes abrupt horizontal and vertical
        variations while preserving the main sensitivity
        structure.

        Parameters
        ----------
        saliency : torch.Tensor
            Saliency map with shape (B, H, W).

        Returns
        -------
        torch.Tensor
            Smoothness loss.
        """

        if saliency.ndim != 3:
            raise ValueError(
                "Saliency must have shape (B, H, W)."
            )

        if (
            saliency.shape[-1] < 2
            or saliency.shape[-2] < 2
        ):
            return saliency.new_tensor(0.0)

        horizontal = (
            saliency[:, :, 1:]
            -
            saliency[:, :, :-1]
        )

        vertical = (
            saliency[:, 1:, :]
            -
            saliency[:, :-1, :]
        )

        loss = (
            horizontal.abs().mean()
            +
            vertical.abs().mean()
        )

        return loss

    # =========================================================
    # MUTUAL-INFORMATION-INSPIRED OBJECTIVE
    # =========================================================

    def mutual_information_loss(
        self,
        features,
        labels
    ):
        """
        Mutual-information-inspired discriminative objective.

        The objective encourages the learned representation to
        retain class-discriminative information.

        A normalized class-centroid separation formulation is
        used here. Within-class compactness is minimized while
        between-class separation is maximized.

        This provides a differentiable representation-level
        objective without introducing an additional neural
        estimator.

        Parameters
        ----------
        features : torch.Tensor
            Feature representation (B, D).

        labels : torch.Tensor
            Ground-truth class labels (B,).

        Returns
        -------
        torch.Tensor
            Representation discrimination loss.
        """

        if features.ndim != 2:
            raise ValueError(
                "features must have shape (B, D)."
            )

        if labels.ndim != 1:
            labels = labels.view(-1)

        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                "features and labels must contain the same "
                "number of samples."
            )

        unique_classes = torch.unique(
            labels
        )

        if unique_classes.numel() < 2:
            return features.new_tensor(0.0)

        # Normalize the representation.
        normalized = F.normalize(
            features,
            p=2,
            dim=1
        )

        centroids = []
        within_class = features.new_tensor(0.0)

        valid_classes = []

        for cls in unique_classes:

            mask = labels == cls

            if mask.sum() == 0:
                continue

            class_features = normalized[
                mask
            ]

            centroid = F.normalize(
                class_features.mean(dim=0, keepdim=True),
                p=2,
                dim=1
            ).squeeze(0)

            centroids.append(
                centroid
            )

            valid_classes.append(cls)

            # Within-class dispersion.
            distances = (
                1.0
                -
                torch.sum(
                    class_features * centroid.unsqueeze(0),
                    dim=1
                )
            )

            within_class = (
                within_class
                +
                distances.mean()
            )

        if len(centroids) < 2:
            return features.new_tensor(0.0)

        centroids = torch.stack(
            centroids,
            dim=0
        )

        within_class = (
            within_class
            /
            len(centroids)
        )

        # Pairwise centroid similarity.
        similarity = torch.mm(
            centroids,
            centroids.t()
        )

        mask = ~torch.eye(
            similarity.shape[0],
            dtype=torch.bool,
            device=similarity.device
        )

        between_similarity = (
            similarity[mask].mean()
        )

        # Lower within-class dispersion and lower
        # between-class similarity are desirable.
        loss = (
            within_class
            +
            between_similarity
        )

        return loss

    # =========================================================
    # CONFIDENCE REGULARIZATION
    # =========================================================

    @staticmethod
    def confidence_regularization(
        logits,
        labels
    ):
        """
        Confidence regularization.

        The objective discourages excessive confidence when the
        predicted class is incorrect while retaining confidence
        for correctly classified samples.

        This is implemented using the difference between
        predicted confidence and correctness.

        Parameters
        ----------
        logits : torch.Tensor
            Class logits (B, K).

        labels : torch.Tensor
            Ground-truth labels (B,).

        Returns
        -------
        torch.Tensor
            Confidence regularization loss.
        """

        if logits.ndim != 2:
            raise ValueError(
                "logits must have shape (B, K)."
            )

        probabilities = F.softmax(
            logits,
            dim=1
        )

        confidence, prediction = (
            probabilities.max(dim=1)
        )

        correct = (
            prediction == labels
        ).float()

        # Correct predictions should support confidence,
        # whereas incorrect predictions should not become
        # excessively confident.
        loss = (
            correct * (1.0 - confidence)
            +
            (1.0 - correct) * confidence
        )

        return loss.mean()

    # =========================================================
    # COMPLETE TRAINING OBJECTIVE
    # =========================================================

    def compute_training_loss(
        self,
        images,
        labels,
        beta=0.10,
        lambda2=0.10
    ):
        """
        Compute the complete ASGSR CNN training objective.

        Objective structure:

            L_total =
                L_cls
                + beta * L_saliency
                + L_MI
                + lambda2 * L_conf

        where:

            L_cls
                = classification loss

            L_saliency
                = spatial saliency smoothness

            L_MI
                = representation-level
                  discriminative objective

            L_conf
                = confidence regularization

        Parameters
        ----------
        images : torch.Tensor
            MRI batch.

        labels : torch.Tensor
            Ground-truth labels.

        beta : float
            Saliency smoothness weight.
            Default = 0.10.

        lambda2 : float
            Confidence regularization weight.
            Default = 0.10.

        Returns
        -------
        dict
            Complete loss dictionary.
        """

        if beta < 0:
            raise ValueError(
                "beta must be non-negative."
            )

        if lambda2 < 0:
            raise ValueError(
                "lambda2 must be non-negative."
            )

        # -----------------------------------------------------
        # Forward pass
        # -----------------------------------------------------

        logits = self.forward(
            images
        )

        # -----------------------------------------------------
        # Classification objective
        # -----------------------------------------------------

        classification_loss = F.cross_entropy(
            logits,
            labels
        )

        # -----------------------------------------------------
        # Representation
        # -----------------------------------------------------

        features = self._last_feature_vector

        if features is None:
            raise RuntimeError(
                "Feature representation was not generated."
            )

        # -----------------------------------------------------
        # Saliency
        #
        # Use a differentiable predicted-class response for
        # the saliency regularization term.
        # -----------------------------------------------------

        predicted_class = torch.argmax(
            logits.detach(),
            dim=1
        )

        batch_indices = torch.arange(
            logits.shape[0],
            device=logits.device
        )

        predicted_scores = logits[
            batch_indices,
            predicted_class
        ]

        saliency_grad = torch.autograd.grad(
            predicted_scores.sum(),
            images,
            create_graph=True,
            retain_graph=True,
            allow_unused=False
        )[0]

        saliency = saliency_grad.abs().max(
            dim=1
        ).values

        maximum = (
            saliency
            .flatten(1)
            .amax(dim=1)
            .view(-1, 1, 1)
        )

        saliency = torch.where(
            maximum > 1e-8,
            saliency / maximum.clamp_min(1e-8),
            torch.zeros_like(saliency)
        )

        saliency_loss = (
            self.saliency_smoothness_loss(
                saliency
            )
        )

        # -----------------------------------------------------
        # Mutual-information-inspired representation objective
        # -----------------------------------------------------

        mi_loss = (
            self.mutual_information_loss(
                features,
                labels
            )
        )

        # -----------------------------------------------------
        # Confidence regularization
        # -----------------------------------------------------

        confidence_loss = (
            self.confidence_regularization(
                logits,
                labels
            )
        )

        # -----------------------------------------------------
        # Total objective
        # -----------------------------------------------------

        total_loss = (
            classification_loss
            +
            beta * saliency_loss
            +
            mi_loss
            +
            lambda2 * confidence_loss
        )

        return {
            "total": total_loss,
            "classification": classification_loss,
            "saliency": saliency_loss,
            "mutual_information": mi_loss,
            "confidence": confidence_loss
        }


# =============================================================
# SIMPLE SELF-TEST
# =============================================================

if __name__ == "__main__":

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = SaliencyCNN(
        num_classes=4,
        in_channels=4
    ).to(device)

    images = torch.randn(
        4,
        4,
        224,
        224,
        device=device
    )

    labels = torch.tensor(
        [0, 1, 2, 3],
        device=device
    )

    # ---------------------------------------------------------
    # Forward test
    # ---------------------------------------------------------

    logits = model(
        images
    )

    print(
        "Logits shape:",
        tuple(logits.shape)
    )

    # ---------------------------------------------------------
    # Loss test
    # ---------------------------------------------------------

    losses = model.compute_training_loss(
        images,
        labels,
        beta=0.10,
        lambda2=0.10
    )

    print(
        "Total loss:",
        float(losses["total"].detach())
    )

    print(
        "Classification loss:",
        float(losses["classification"].detach())
    )

    print(
        "Saliency loss:",
        float(losses["saliency"].detach())
    )

    print(
        "Mutual-information loss:",
        float(losses["mutual_information"].detach())
    )

    print(
        "Confidence loss:",
        float(losses["confidence"].detach())
    )

    # ---------------------------------------------------------
    # Saliency test
    # ---------------------------------------------------------

    saliency, predicted_class = (
        model.compute_saliency(
            images[:1]
        )
    )

    print(
        "Saliency shape:",
        tuple(saliency.shape)
    )

    print(
        "Predicted class:",
        predicted_class.tolist()
    )

    print(
        "Saliency range:",
        float(saliency.min()),
        "to",
        float(saliency.max())
    )
