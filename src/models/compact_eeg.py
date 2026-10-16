import torch
import torch.nn as nn


class CompactEEG(nn.Module):
    def __init__(
        self,
        n_channels: int = 64,
        n_times: int = 120,
        patch_size: int = 10,
        d_model: int = 192,
        output_dim: int = 1536,
    ):
        super().__init__()

        assert n_times % patch_size == 0, (
            f"n_times={n_times} must be divisible by patch_size={patch_size}"
        )

        self.n_channels = n_channels
        self.n_times = n_times
        self.patch_size = patch_size
        self.d_model = d_model

        self.n_patches = n_times // patch_size

        self.patch_projection = nn.Linear(
            patch_size,
            d_model,
        )

        self.output_head = nn.Linear(
            d_model,
            output_dim,
        )

    def forward(self, x):
        """
        x: (B, C, T)
        """

        B, C, T = x.shape

        if C != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {C}"
            )

        if T != self.n_times:
            raise ValueError(
                f"Expected {self.n_times} time samples, got {T}"
            )

        # (B, C, T)
        # ->
        # (B, C, n_patches, patch_size)
        x = x.reshape(
            B,
            C,
            self.n_patches,
            self.patch_size,
        )

        # Each 10-sample temporal patch becomes
        # a d_model-dimensional token.
        x = self.patch_projection(x)

        # (B, C, n_patches, d_model)

        # For now, simply average all channel/time tokens.
        x = x.mean(dim=(1, 2))

        # (B, d_model)

        x = self.output_head(x)

        # (B, output_dim)

        return x