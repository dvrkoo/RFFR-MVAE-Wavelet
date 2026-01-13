import os
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.vision_transformer import VisionTransformer

from models.model_mae import mae_vit_base_patch16 as standard_mae
from models.model_mae_vae import mae_vae_vit_base_patch16 as mae_vae
import sys
from utils.wavelet_utils import (
    extract_all_wavelet_subbands,
    create_fused_wavelet_representation,
    rgb_to_wavelet,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../rffr_generative"))
from models.model_vae import vae_base
from models.model_wavelet_vae import wavelet_vae_base

from configs.config import config

# ViT Base configuration
VIT_CONFIG = {
    "patch_size": 16,
    "embed_dim": 768,
    "depth": 12,
    "num_heads": 12,
    "mlp_ratio": 4,
    "qkv_bias": True,
    "norm_layer": partial(nn.LayerNorm, eps=1e-6),
}


class BlockClassifier(VisionTransformer):
    """Standard ViT classifier with optional pretrained weight loading."""

    def __init__(self, load_pretrained=True, **kwargs):
        super(BlockClassifier, self).__init__(**kwargs)
        if (
            load_pretrained
            and config.pretrained_weights
            and os.path.exists(config.pretrained_weights)
        ):
            self.load_state_dict(torch.load(config.pretrained_weights))

    def forward(self, x, patch_id=None):
        D = self.embed_dim
        B = x.shape[0]

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        if patch_id is not None:
            # partial: [Batchsize, L_keep]
            # indexes in partial are the ones to keep.
            patch_id += 1  # This is to skip over the class token.
            patch_id = patch_id.to(x.device).type(torch.int64)
            x_masked = torch.gather(
                x, dim=1, index=patch_id.unsqueeze(-1).repeat(1, 1, D)
            )
            x = x_masked

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x[:, 0]


class AdaptivePatchEmbed(nn.Module):
    """
    Adaptive patch embedding that dynamically adjusts patch size based on input dimensions.
    Maintains approximately the same number of patches regardless of input size.
    """

    def __init__(self, target_patches=196, embed_dim=768, in_chans=3):
        super().__init__()
        self.target_patches = target_patches
        self.embed_dim = embed_dim
        self.in_chans = in_chans

    def forward(self, x):
        B, C, H, W = x.shape

        # Calculate optimal patch size to get close to target_patches
        total_pixels = H * W
        optimal_patch_area = total_pixels / self.target_patches
        patch_size = max(1, int(math.sqrt(optimal_patch_area)))

        # Find largest patch size that divides dimensions evenly
        for ps in range(patch_size, 0, -1):
            if H % ps == 0 and W % ps == 0:
                patch_size = ps
                break

        # Use patch_size=1 if no even division found
        if patch_size == 0:
            patch_size = 1

        # Create dynamic conv layer
        conv = nn.Conv2d(
            C, self.embed_dim, kernel_size=patch_size, stride=patch_size
        ).to(x.device)

        # Initialize weights
        with torch.no_grad():
            nn.init.xavier_uniform_(conv.weight.view([conv.weight.shape[0], -1]))
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

        # Apply patch embedding
        x = conv(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size

        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        return x, (num_patches_h, num_patches_w)


def _get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Generate 2D sinusoidal positional embeddings."""
    grid_h, grid_w = int(grid_size[0]), int(grid_size[1])
    assert embed_dim % 2 == 0

    # Use half embedding dim for each spatial dimension
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, np.arange(grid_h))
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, np.arange(grid_w))

    emb = np.concatenate(
        [
            emb_h[:, None, :].repeat(grid_w, axis=1),
            emb_w[None, :, :].repeat(grid_h, axis=0),
        ],
        axis=-1,
    )
    emb = emb.reshape([grid_h * grid_w, embed_dim])

    if cls_token:
        emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)

    return emb


def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Generate 1D sinusoidal positional embeddings."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class AdaptiveViT(VisionTransformer):
    """
    Vision Transformer with adaptive patch embedding for variable input sizes.
    Maintains consistent transformer architecture while adapting to different image resolutions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Replace fixed patch embedding with adaptive one
        in_chans = kwargs.get(
            "in_chans", 3
        )  # Support variable input channels (3 for RGB, 12 for wavelets)
        self.adaptive_patch_embed = AdaptivePatchEmbed(
            target_patches=196,  # Same as standard ViT
            embed_dim=self.embed_dim,
            in_chans=in_chans,
        )

        # Remove the fixed patch embedding and positional embeddings
        delattr(self, "patch_embed")
        if hasattr(self, "pos_embed"):
            delattr(self, "pos_embed")

    def _interpolate_pos_embed(self, grid_size):
        """Create positional embeddings for any grid size."""
        h, w = grid_size
        pos_embed = _get_2d_sincos_pos_embed(self.embed_dim, (h, w), cls_token=False)
        return torch.from_numpy(pos_embed).float().unsqueeze(0)

    def forward(self, x, patch_id=None):
        B = x.shape[0]

        # Adaptive patch embedding
        x, grid_size = self.adaptive_patch_embed(x)

        # Generate positional encoding for current grid size
        pos_embed = self._interpolate_pos_embed(grid_size).to(x.device)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional encoding (cls token gets zero embedding)
        cls_pos = torch.zeros(1, 1, self.embed_dim, device=x.device)
        full_pos_embed = torch.cat([cls_pos, pos_embed], dim=1)
        x = x + full_pos_embed

        x = self.pos_drop(x)

        # Apply patch selection if provided
        if patch_id is not None:
            D = self.embed_dim
            patch_id_with_cls = patch_id + 1  # Skip class token
            patch_id_with_cls = patch_id_with_cls.to(x.device).type(torch.int64)

            # Keep class token + selected patches
            cls_token = x[:, :1, :]  # [B, 1, D]
            selected_patches = torch.gather(
                x[:, 1:, :],
                dim=1,
                index=patch_id_with_cls.unsqueeze(-1).repeat(1, 1, D),
            )  # [B, num_selected, D]

            x = torch.cat([cls_token, selected_patches], dim=1)

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x[:, 0]  # Return class token


class WaveletSubbandClassifier(nn.Module):
    """CNN-based classifier for wavelet subbands that can handle any input size."""

    def __init__(self, embed_dim=768, **kwargs):
        super().__init__()

        # CNN backbone that adapts to input size
        self.backbone = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling to fixed size
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256, embed_dim), nn.ReLU(), nn.Dropout(0.1)
        )

    def forward(self, x, patch_id=None):
        # patch_id is ignored for CNN-based classifier
        x = self.backbone(x)
        x = self.classifier(x)
        return x


class FourBranchWaveletClassifier(nn.Module):
    """
    4-branch wavelet-only classifier that processes each wavelet subband independently.
    Uses either AdaptiveViT or CNN-based classifiers based on config.use_adaptive_vit.
    """

    def __init__(self, **kwargs):
        super().__init__()

        embed_dim = kwargs.get("embed_dim", VIT_CONFIG["embed_dim"])

        if config.use_adaptive_vit:
            # Use AdaptiveViT branches for each wavelet subband
            self.branch_LL = AdaptiveViT(**kwargs)  # Approximation
            self.branch_LH = AdaptiveViT(**kwargs)  # Vertical detail
            self.branch_HL = AdaptiveViT(**kwargs)  # Horizontal detail
            self.branch_HH = AdaptiveViT(**kwargs)  # Diagonal detail

            # Load pretrained weights only if configured
            if config.use_imagenet_pretrain_for_wavelets and config.pretrained_weights:
                self._load_pretrained_weights_for_branches()
            else:
                print(
                    "Using random initialization for wavelet branches (recommended for frequency domain)"
                )
        else:
            # Use CNN-based branches
            self.branch_LL = WaveletSubbandClassifier(embed_dim=embed_dim)
            self.branch_LH = WaveletSubbandClassifier(embed_dim=embed_dim)
            self.branch_HL = WaveletSubbandClassifier(embed_dim=embed_dim)
            self.branch_HH = WaveletSubbandClassifier(embed_dim=embed_dim)

        # Fusion layer to combine features from all 4 branches
        self.fusion = nn.Sequential(
            nn.Linear(4 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
        )

    def _load_pretrained_weights_for_branches(self):
        """Load pretrained weights for all wavelet branches."""
        try:
            pretrained_state = torch.load(config.pretrained_weights)

            # Load weights for each branch, excluding patch_embed and pos_embed
            for branch_name, branch in [
                ("LL", self.branch_LL),
                ("LH", self.branch_LH),
                ("HL", self.branch_HL),
                ("HH", self.branch_HH),
            ]:
                # Filter out incompatible weights
                filtered_state = {
                    k: v
                    for k, v in pretrained_state.items()
                    if not k.startswith("patch_embed") and not k.startswith("pos_embed")
                }
                branch.load_state_dict(filtered_state, strict=False)
                print(
                    f"Loaded ImageNet pretrained weights for {branch_name} wavelet branch (adaptive)"
                )
        except Exception as e:
            print(
                f"Warning: Could not load pretrained weights for wavelet branches: {e}"
            )

    def forward(self, wavelet_subbands, patch_id=None):
        """
        Forward pass for 4-branch wavelet classifier.

        Args:
            wavelet_subbands: Dict with keys 'LL', 'LH', 'HL', 'HH' containing subband tensors
            patch_id: Patch selection indices for all branches

        Returns:
            Tensor: Fused features from all 4 wavelet branches (B, embed_dim)
        """
        # Process each subband with its dedicated branch
        features_LL = self.branch_LL(wavelet_subbands["LL"], patch_id=patch_id)
        features_LH = self.branch_LH(wavelet_subbands["LH"], patch_id=patch_id)
        features_HL = self.branch_HL(wavelet_subbands["HL"], patch_id=patch_id)
        features_HH = self.branch_HH(wavelet_subbands["HH"], patch_id=patch_id)

        # Concatenate all branch features
        combined_features = torch.cat(
            [features_LL, features_LH, features_HL, features_HH], dim=1
        )

        # Apply fusion layer
        fused_features = self.fusion(combined_features)

        return fused_features


class ResidualGenerator(nn.Module):
    """Generates residuals using MAE-based inpainting with optional wavelet processing."""

    def __init__(self):
        super().__init__()

        if not config.mae_path or not os.path.exists(config.mae_path):
            raise FileNotFoundError(
                f"Generative model checkpoint not found at: {config.mae_path}. "
                "Please provide a valid path in the config."
            )

        print(f"Loading generative model checkpoint from: {config.mae_path}")
        ckpt = torch.load(config.mae_path, map_location="cpu", weights_only=False)

        if "config" in ckpt:
            gen_config = ckpt["config"]
            generator_type = gen_config.get(
                "generator_type", config.generative_model_type
            )
            vae_latent_dim = gen_config.get("vae_latent_dim", config.vae_latent_dim)
            vae_base_channels = gen_config.get(
                "vae_base_channels", config.vae_base_channels
            )
            wavelet_vae_wavelet_type = gen_config.get("wavelet_vae_wavelet_type", "db4")
            print(
                f"Auto-loaded config from checkpoint: generator_type={generator_type}, latent_dim={vae_latent_dim}, base_channels={vae_base_channels}"
            )
        else:
            generator_type = config.generative_model_type
            vae_latent_dim = config.vae_latent_dim
            vae_base_channels = config.vae_base_channels
            wavelet_vae_wavelet_type = "db4"
            print(
                f"Warning: Checkpoint has no config metadata. Using manual config from config.py"
            )
            print(
                f"Manual config: generator_type={generator_type}, latent_dim={vae_latent_dim}, base_channels={vae_base_channels}"
            )

        print(
            f"Initializing generative model of type: '{generator_type}' for residual generation."
        )

        if generator_type == "vae":
            generative_model = vae_base(
                latent_dim=vae_latent_dim, base_channels=vae_base_channels
            ).cuda()
            print(
                f"Loaded VAE with latent_dim={vae_latent_dim}, base_channels={vae_base_channels}"
            )
            self.use_vae = True
        elif generator_type == "wavelet_vae":
            generative_model = wavelet_vae_base(
                latent_dim=vae_latent_dim,
                base_channels=vae_base_channels,
                wavelet_type=wavelet_vae_wavelet_type,
            ).cuda()
            print(
                f"Loaded WaveletVAE with latent_dim={vae_latent_dim}, base_channels={vae_base_channels}, wavelet={wavelet_vae_wavelet_type}"
            )
            self.use_vae = True
        elif generator_type == "mae_vae":
            generative_model = mae_vae(
                vae_latent_dim=vae_latent_dim,
                freeze_encoder=True,
            ).cuda()
            print(f"Loaded MAE-VAE with latent dimension: {vae_latent_dim}")
            self.use_vae = False
        else:
            generative_model = standard_mae().cuda()
            print("Loaded standard MAE.")
            self.use_vae = False

        generative_model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"Successfully loaded model weights from checkpoint")

        self.inpainter = generative_model
        # The `magic` and `full_patches` attributes remain the same.
        # Index offsets for a 3x3 region on a 14x14 grid
        self.magic = torch.Tensor([0, 1, 2, 14, 15, 16, 28, 29, 30])
        self.full_patches = self._block_to_patch(torch.arange(16).unsqueeze(0))

    def _block_to_patch(self, idx, test=False):
        """Convert block indices to patch indices."""
        # This function remains UNCHANGED.
        bs, blk = idx.shape
        if test:
            return self.full_patches.repeat(bs, 1)

        w, h = 3 * (torch.div(idx, 4, rounding_mode="floor")) + 1, 3 * (idx % 4) + 1
        idx = w * 14 + h

        idx = idx.repeat(1, 9).reshape(bs, -1, blk).transpose(1, 2)
        idx = idx + self.magic
        idx = idx.reshape(bs, -1)
        return idx

    def forward(self, rgb_01, test=False):
        """Generate residuals from RGB input."""
        if self.use_vae:
            # VAE: Full-image reconstruction without block masking
            with torch.no_grad():
                res = self.inpainter.reconstruct(rgb_01, deterministic=True)

            # VAE doesn't use block/patch masking, so return full patches
            block_id = torch.arange(16).unsqueeze(0).repeat(rgb_01.shape[0], 1)
            patch_id = self._block_to_patch(block_id, test=True)
            num_blocks = 16

            return res, patch_id, num_blocks
        else:
            # MAE/MAE-VAE: Block-based reconstruction with masking
            if test:
                block_id = None
            else:
                blocks = []
                for i in range(16):
                    if np.random.rand(1) < 0.25:
                        blocks.append(i)
                block_id = (
                    None
                    if len(blocks) == 0
                    else torch.LongTensor(blocks)
                    .unsqueeze(0)
                    .repeat(rgb_01.shape[0], 1)
                )

            with torch.no_grad():
                if config.wavelet_residual_branch:
                    spatial_res, reconstructed_img, block_id = (
                        self.inpainter.patch_by_patch_DIFF(
                            rgb_01,
                            block_id=block_id,
                            test=test,
                            return_reconstructed=True,
                        )
                    )

                    original_wavelets = rgb_to_wavelet(rgb_01)
                    reconstructed_wavelets = rgb_to_wavelet(reconstructed_img)
                    wavelet_res = original_wavelets - reconstructed_wavelets
                elif config.wavelet_dual_branch_mode:
                    original_wavelets, wavelet_residuals, block_id = (
                        self.inpainter.patch_by_patch_DIFF(
                            rgb_01,
                            block_id=block_id,
                            test=test,
                            return_wavelet_dual=True,
                        )
                    )
                elif config.wavelet_only_mode:
                    res, block_id = self.inpainter.patch_by_patch_DIFF(
                        rgb_01,
                        block_id=block_id,
                        test=test,
                        return_wavelet_residuals=True,
                    )
                else:
                    res, block_id = self.inpainter.patch_by_patch_DIFF(
                        rgb_01, block_id=block_id, test=test
                    )

            # Handle different wavelet architectures
            if config.wavelet_residual_branch:
                res_outputs = {
                    "spatial_residuals": spatial_res,
                    "wavelet_residuals": wavelet_res,
                }
            elif config.wavelet_dual_branch_mode:
                res_outputs = {
                    "original_wavelets": original_wavelets,
                    "wavelet_residuals": wavelet_residuals,
                }
            elif config.wavelet_only_mode:
                res_outputs = res
            elif config.use_wavelets and config.four_branch_wavelet:
                wavelet_subbands = extract_all_wavelet_subbands(res)
                res_outputs = {"wavelet_subbands": wavelet_subbands}
            else:
                res_outputs = res

            patch_id = self._block_to_patch(block_id, test=test)

            num_blocks = block_id.shape[1] if block_id is not None else 16
            return res_outputs, patch_id, num_blocks


class DeepfakeDetector(nn.Module):
    """Multi-branch deepfake detection model supporting various wavelet architectures."""

    def __init__(self):
        super().__init__()

        # Architecture selection based on config
        if config.wavelet_residual_branch:
            # 3-branch architecture: RGB + Spatial Residuals + Wavelet Residuals
            print(
                "Initializing 3-branch architecture (RGB + spatial residuals + wavelet residuals)"
            )

            self.backbone_1 = BlockClassifier(**VIT_CONFIG)  # RGB branch
            self.backbone_2 = BlockClassifier(
                load_pretrained=True, **VIT_CONFIG
            )  # Spatial residual branch

            # Wavelet residual branch (12 channels)
            vit_config_12ch = VIT_CONFIG.copy()
            vit_config_12ch["in_chans"] = 12
            self.backbone_3 = AdaptiveViT(**vit_config_12ch)

            # Load pretrained if configured
            if (
                config.use_imagenet_pretrain_for_wavelets
                and config.pretrained_weights
                and os.path.exists(config.pretrained_weights)
            ):
                self._load_pretrained_for_wavelet_residual_branch()
            else:
                print(
                    "Using random initialization for wavelet residual branch (recommended for frequency domain)"
                )

            # Nullify other branch options
            self.backbone_wavelet_original = None
            self.backbone_wavelet_residual = None
            self.four_branch_classifier = None

            classifier_input_dim = VIT_CONFIG["embed_dim"] * 3  # 768 * 3 = 2304

        elif config.wavelet_dual_branch_mode:
            # 2-branch wavelet architecture: Original wavelets + Wavelet residuals
            print(
                "Initializing 2-branch wavelet architecture (original wavelets + wavelet residuals)"
            )

            self.backbone_1 = None
            self.backbone_2 = None
            self.four_branch_classifier = None

            # Create two 12-channel AdaptiveViT branches
            vit_config_12ch = VIT_CONFIG.copy()
            vit_config_12ch["in_chans"] = 12
            self.backbone_wavelet_original = AdaptiveViT(**vit_config_12ch)
            self.backbone_wavelet_residual = AdaptiveViT(**vit_config_12ch)
            self.backbone_3 = None

            # Load pretrained weights if configured (excluding patch_embed)
            if (
                config.use_imagenet_pretrain_for_wavelets
                and config.pretrained_weights
                and os.path.exists(config.pretrained_weights)
            ):
                self._load_pretrained_for_dual_branch()
            else:
                print(
                    "Using random initialization for dual wavelet branches (recommended for frequency domain)"
                )

            classifier_input_dim = (
                VIT_CONFIG["embed_dim"] * 2
            )  # Two branches: 768 * 2 = 1536

        elif config.wavelet_only_mode:
            # Wavelet-only architecture: Single branch processing 12-channel wavelet residuals
            print("Initializing wavelet-only architecture (single branch, 12 channels)")

            self.backbone_1 = None
            self.backbone_2 = None
            self.four_branch_classifier = None
            self.backbone_wavelet_original = None
            self.backbone_wavelet_residual = None

            # Create 12-channel AdaptiveViT for wavelet processing
            vit_config_12ch = VIT_CONFIG.copy()
            vit_config_12ch["in_chans"] = 12
            self.backbone_3 = AdaptiveViT(**vit_config_12ch)

            # Load pretrained weights if configured (excluding patch_embed)
            if (
                config.use_imagenet_pretrain_for_wavelets
                and config.pretrained_weights
                and os.path.exists(config.pretrained_weights)
            ):
                self._load_pretrained_for_wavelet_only_branch()
            else:
                print(
                    "Using random initialization for wavelet-only branch (recommended for frequency domain)"
                )

            classifier_input_dim = VIT_CONFIG["embed_dim"]  # Single branch: 768

        elif (
            config.use_wavelets
            and config.separate_wavelet_branch
            and not config.four_branch_wavelet
        ):
            # 3-branch architecture: RGB + Residuals + Wavelets
            self.backbone_1 = BlockClassifier(**VIT_CONFIG)
            self.backbone_2 = BlockClassifier(load_pretrained=True, **VIT_CONFIG)
            self.backbone_wavelet_original = None
            self.backbone_wavelet_residual = None

            if config.use_adaptive_vit:
                self.backbone_3 = AdaptiveViT(**VIT_CONFIG)
                self._load_pretrained_for_adaptive_wavelet_branch()
            else:
                self.backbone_3 = BlockClassifier(
                    load_pretrained=config.use_imagenet_pretrain_for_wavelets,
                    **VIT_CONFIG,
                )

            self.four_branch_classifier = None
            classifier_input_dim = VIT_CONFIG["embed_dim"] * 3

        elif config.use_wavelets and config.four_branch_wavelet:
            # 5-branch architecture
            self.backbone_1 = BlockClassifier(**VIT_CONFIG)
            self.backbone_2 = None
            self.backbone_3 = None
            self.backbone_wavelet_original = None
            self.backbone_wavelet_residual = None
            self.four_branch_classifier = FourBranchWaveletClassifier(**VIT_CONFIG)
            classifier_input_dim = VIT_CONFIG["embed_dim"] * 2
        else:
            # 2-branch architecture: RGB + Residuals
            self.backbone_1 = BlockClassifier(**VIT_CONFIG)
            self.backbone_2 = BlockClassifier(load_pretrained=True, **VIT_CONFIG)
            self.backbone_3 = None
            self.backbone_wavelet_original = None
            self.backbone_wavelet_residual = None
            self.four_branch_classifier = None
            classifier_input_dim = VIT_CONFIG["embed_dim"] * 2

        self.classifier = nn.Linear(classifier_input_dim, 2)

    def _load_pretrained_for_dual_branch(self):
        """Load pretrained weights for both dual-branch wavelet branches if configured."""
        try:
            pretrained_state = torch.load(config.pretrained_weights)
            filtered_state = {
                k: v
                for k, v in pretrained_state.items()
                if not k.startswith("patch_embed") and not k.startswith("pos_embed")
            }
            self.backbone_wavelet_original.load_state_dict(filtered_state, strict=False)
            self.backbone_wavelet_residual.load_state_dict(filtered_state, strict=False)
            print(
                "Loaded ImageNet pretrained weights for dual wavelet branches (excluding patch_embed)"
            )
        except Exception as e:
            print(
                f"Warning: Could not load pretrained weights for dual wavelet branches: {e}"
            )

    def _load_pretrained_for_wavelet_residual_branch(self):
        """Load pretrained weights for wavelet residual branch if configured."""
        try:
            pretrained_state = torch.load(config.pretrained_weights)
            filtered_state = {
                k: v
                for k, v in pretrained_state.items()
                if not k.startswith("patch_embed") and not k.startswith("pos_embed")
            }
            self.backbone_3.load_state_dict(filtered_state, strict=False)
            print(
                "Loaded ImageNet pretrained weights for wavelet residual branch (excluding patch_embed)"
            )
        except Exception as e:
            print(
                f"Warning: Could not load pretrained weights for wavelet residual branch: {e}"
            )

    def _load_pretrained_for_wavelet_only_branch(self):
        """Load pretrained weights for wavelet-only branch if configured."""
        try:
            pretrained_state = torch.load(config.pretrained_weights)
            filtered_state = {
                k: v
                for k, v in pretrained_state.items()
                if not k.startswith("patch_embed") and not k.startswith("pos_embed")
            }
            self.backbone_3.load_state_dict(filtered_state, strict=False)
            print(
                "Loaded ImageNet pretrained weights for wavelet-only branch (excluding patch_embed)"
            )
        except Exception as e:
            print(
                f"Warning: Could not load pretrained weights for wavelet-only branch: {e}"
            )

    def _load_pretrained_for_adaptive_wavelet_branch(self):
        """Load pretrained weights for adaptive wavelet branch if configured."""
        if (
            config.use_imagenet_pretrain_for_wavelets
            and config.pretrained_weights
            and os.path.exists(config.pretrained_weights)
        ):
            try:
                pretrained_state = torch.load(config.pretrained_weights)
                # Filter out incompatible weights since AdaptiveViT uses different patch embedding
                filtered_state = {
                    k: v
                    for k, v in pretrained_state.items()
                    if not k.startswith("patch_embed") and not k.startswith("pos_embed")
                }
                self.backbone_3.load_state_dict(filtered_state, strict=False)
                print(
                    "Loaded ImageNet pretrained weights for wavelet branch (adaptive)"
                )
            except Exception as e:
                print(
                    f"Warning: Could not load pretrained weights for wavelet branch: {e}"
                )
        else:
            print(
                "Using random initialization for wavelet branch (recommended for frequency domain)"
            )

    def forward(self, res_data, rgb_norm, patch_id):
        """Forward pass through the detection network."""

        # Handle different architectures based on config
        if config.wavelet_residual_branch:
            # 3-branch architecture: RGB + Spatial Residuals + Wavelet Residuals
            if (
                isinstance(res_data, dict)
                and "spatial_residuals" in res_data
                and "wavelet_residuals" in res_data
            ):
                spatial_residuals = res_data["spatial_residuals"]
                wavelet_residuals = res_data["wavelet_residuals"]

                # Process each branch
                rgb_features = self.backbone_1(rgb_norm, patch_id=patch_id)
                spatial_features = self.backbone_2(spatial_residuals, patch_id=patch_id)
                wavelet_features = self.backbone_3(wavelet_residuals, patch_id=patch_id)

                # Combine features: 768 + 768 + 768 = 2304
                feature = torch.cat(
                    [rgb_features, spatial_features, wavelet_features], dim=1
                )
            else:
                raise ValueError(
                    "Expected res_data dict with 'spatial_residuals' and 'wavelet_residuals' keys for wavelet_residual_branch mode"
                )

        elif config.wavelet_dual_branch_mode:
            # 2-branch wavelet architecture: Original wavelets + Wavelet residuals
            if (
                isinstance(res_data, dict)
                and "original_wavelets" in res_data
                and "wavelet_residuals" in res_data
            ):
                original_wavelets = res_data["original_wavelets"]
                wavelet_residuals = res_data["wavelet_residuals"]

                # Process original wavelets through first branch
                original_wavelet_features = self.backbone_wavelet_original(
                    original_wavelets, patch_id=patch_id
                )

                # Process wavelet residuals through second branch
                residual_wavelet_features = self.backbone_wavelet_residual(
                    wavelet_residuals, patch_id=patch_id
                )

                # Combine features: 768 + 768 = 1536
                feature = torch.cat(
                    [original_wavelet_features, residual_wavelet_features], dim=1
                )
            else:
                raise ValueError(
                    "Expected res_data dict with 'original_wavelets' and 'wavelet_residuals' keys for dual-branch mode"
                )

        elif config.wavelet_only_mode:
            # Wavelet-only architecture: Single branch processing 12-channel wavelet residuals
            wavelet_features = self.backbone_3(res_data, patch_id=patch_id)
            feature = wavelet_features

        elif (
            config.use_wavelets
            and config.separate_wavelet_branch
            and not config.four_branch_wavelet
        ):
            # 3-branch architecture: RGB + Residuals + Wavelets
            rgb_features = self.backbone_1(rgb_norm, patch_id=patch_id)
            residual_features = self.backbone_2(res_data, patch_id=patch_id)

            # Create fused wavelet representation and process through wavelet branch
            fused_wavelets = create_fused_wavelet_representation(res_data)
            wavelet_features = self.backbone_3(fused_wavelets, patch_id=patch_id)

            # Combine all three feature sets
            feature = torch.cat(
                [rgb_features, residual_features, wavelet_features], dim=1
            )

        elif config.use_wavelets and config.four_branch_wavelet:
            # 5-branch RGB+wavelet architecture
            if isinstance(res_data, dict) and "wavelet_subbands" in res_data:
                wavelet_subbands = res_data["wavelet_subbands"]

                # Process RGB through standard ViT branch
                rgb_features = self.backbone_1(rgb_norm, patch_id=patch_id)

                # Process all 4 wavelet subbands through dedicated classifier
                wavelet_features = self.four_branch_classifier(
                    wavelet_subbands, patch_id=patch_id
                )

                # Combine RGB + wavelet features
                feature = torch.cat([rgb_features, wavelet_features], dim=1)
            else:
                # Fallback to 2-branch if wavelet data unavailable
                rgb_features = self.backbone_1(rgb_norm, patch_id=patch_id)
                residual_features = self.backbone_2(res_data, patch_id=patch_id)
                feature = torch.cat([rgb_features, residual_features], dim=1)
        else:
            # 2-branch architecture: RGB + Residuals
            rgb_features = self.backbone_1(rgb_norm, patch_id=patch_id)
            residual_features = self.backbone_2(res_data, patch_id=patch_id)
            feature = torch.cat([rgb_features, residual_features], dim=1)

        output = self.classifier(feature)
        return feature, output


class RFFRL(nn.Module):
    """Main RFFR model combining residual generation and deepfake detection."""

    def __init__(self):
        super().__init__()
        self.rg = ResidualGenerator()
        self.dd = DeepfakeDetector()

    def forward(self, rgb_01, rgb_norm, test=False, current_iter=None):
        """Forward pass through the complete RFFR pipeline."""
        res, patch_id, num_blocks = self.rg(rgb_01, test=test)
        feature, output = self.dd(res, rgb_norm, patch_id)
        return feature, output, num_blocks


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    dummy_01 = torch.rand(2, 3, 224, 224).cuda()
    dummy_norm = torch.randn(2, 3, 224, 224).cuda()
    net = RFFRL().cuda()

    result = net(dummy_01, dummy_norm, test=False)
    print(f"Feature shape: {result[0].shape}")
    print(f"Output shape: {result[1].shape}")
