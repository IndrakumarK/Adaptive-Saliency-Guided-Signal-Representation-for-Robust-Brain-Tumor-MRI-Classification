import torch
import torch.nn as nn
import torch.nn.functional as F


class SaliencyCNN(nn.Module):
    """
    Lightweight CNN for gradient-based saliency estimation

    Architecture:
    - 4 Conv Blocks (32, 64, 128, 256)
    - BatchNorm + ReLU
    - Fully Connected Layers
    """

    def __init__(self, num_classes=4):
        super(SaliencyCNN, self).__init__()

        # -------------------------------------------------
        # Convolutional Blocks
        # -------------------------------------------------
        self.conv1 = self._conv_block(3, 32)
        self.conv2 = self._conv_block(32, 64)
        self.conv3 = self._conv_block(64, 128)
        self.conv4 = self._conv_block(128, 256)

        self.pool = nn.MaxPool2d(2, 2)

        # -------------------------------------------------
        # Fully Connected Layers
        # -------------------------------------------------
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, num_classes)

    # -------------------------------------------------
    # Conv Block
    # -------------------------------------------------
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    # -------------------------------------------------
    # Forward Pass
    # -------------------------------------------------
    def forward(self, x):
        """
        x: (B, C, H, W)
        """

        x = self.pool(self.conv1(x))  # 224 ? 112
        x = self.pool(self.conv2(x))  # 112 ? 56
        x = self.pool(self.conv3(x))  # 56 ? 28
        x = self.pool(self.conv4(x))  # 28 ? 14

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # logits

        return x

    # -------------------------------------------------
    # SALIENCY MAP (GRADIENT-BASED)
    # -------------------------------------------------
    def compute_saliency(self, input_tensor):
        """
        Computes gradient-based saliency map

        input_tensor: (1, C, H, W)
        """
        self.eval()

        input_tensor = input_tensor.clone().detach()
        input_tensor.requires_grad = True

        output = self.forward(input_tensor)

        # Take max class score
        score, _ = torch.max(output, dim=1)

        self.zero_grad()
        score.backward()

        saliency = input_tensor.grad.data.abs()

        # Collapse channel dimension
        saliency, _ = torch.max(saliency, dim=1)

        return saliency.squeeze().cpu().numpy()