import math
import os
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

from configs.config import config
from models.model_mae import mae_vit_base_patch16 as standard_mae
from models.model_mae_vae import mae_vae_vit_base_patch16 as mae_vae
from utils.wavelet_utils import rgb_to_wavelet


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
    """Standard ViT classifier used for RGB and spatial residual branches."""

    def __init__(self, load_pretrained=True, **kwargs):
        super().__init__(**kwargs)
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
            patch_id = patch_id + 1
            patch_id = patch_id.to(x.device).type(torch.int64)
            x = torch.gather(x, dim=1, index=patch_id.unsqueeze(-1).repeat(1, 1, D))

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]


class AdaptivePatchEmbed(nn.Module):
    """Patch embedding used by the 12-channel wavelet residual branch."""

    def __init__(self, target_patches=196, embed_dim=768, in_chans=12):
        super().__init__()
        self.target_patches = target_patches
        self.embed_dim = embed_dim
        self.in_chans = in_chans

    def forward(self, x):
        B, C, H, W = x.shape
        optimal_patch_area = (H * W) / self.target_patches
        patch_size = max(1, int(math.sqrt(optimal_patch_area)))

        for ps in range(patch_size, 0, -1):
            if H % ps == 0 and W % ps == 0:
                patch_size = ps
                break

        conv = nn.Conv2d(
            C, self.embed_dim, kernel_size=patch_size, stride=patch_size
        ).to(x.device)

        with torch.no_grad():
            nn.init.xavier_uniform_(conv.weight.view([conv.weight.shape[0], -1]))
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

        x = conv(x)
        grid_size = (H // patch_size, W // patch_size)
        x = x.flatten(2).transpose(1, 2)
        return x, grid_size


def _get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h, grid_w = int(grid_size[0]), int(grid_size[1])
    assert embed_dim % 2 == 0

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
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    return emb


class AdaptiveViT(VisionTransformer):
    """ViT branch for 12-channel wavelet residual inputs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        in_chans = kwargs.get("in_chans", 12)
        self.adaptive_patch_embed = AdaptivePatchEmbed(
            target_patches=196,
            embed_dim=self.embed_dim,
            in_chans=in_chans,
        )
        delattr(self, "patch_embed")
        if hasattr(self, "pos_embed"):
            delattr(self, "pos_embed")

    def _interpolate_pos_embed(self, grid_size):
        pos_embed = _get_2d_sincos_pos_embed(self.embed_dim, grid_size, cls_token=False)
        return torch.from_numpy(pos_embed).float().unsqueeze(0)

    def forward(self, x, patch_id=None):
        B = x.shape[0]
        x, grid_size = self.adaptive_patch_embed(x)

        pos_embed = self._interpolate_pos_embed(grid_size).to(x.device)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_pos = torch.zeros(1, 1, self.embed_dim, device=x.device)

        x = torch.cat((cls_tokens, x), dim=1)
        x = x + torch.cat([cls_pos, pos_embed], dim=1)
        x = self.pos_drop(x)

        if patch_id is not None:
            D = self.embed_dim
            patch_id = patch_id.to(x.device).type(torch.int64)
            cls_token = x[:, :1, :]
            selected_patches = torch.gather(
                x[:, 1:, :], dim=1, index=patch_id.unsqueeze(-1).repeat(1, 1, D)
            )
            x = torch.cat([cls_token, selected_patches], dim=1)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]


class ResidualGenerator(nn.Module):
    """Generate spatial residuals, and wavelet residuals for 3-branch mode."""

    def __init__(self):
        super().__init__()

        if not config.mae_path or not os.path.exists(config.mae_path):
            raise FileNotFoundError(
                f"Generative model checkpoint not found at: {config.mae_path}. "
                "Please provide a valid path in the config."
            )

        print(f"Loading generative model checkpoint from: {config.mae_path}")
        ckpt = torch.load(config.mae_path, map_location="cpu", weights_only=False)

        checkpoint_config = ckpt.get("config", {})
        generator_type = checkpoint_config.get(
            "generator_type", config.generative_model_type
        )
        vae_latent_dim = checkpoint_config.get(
            "vae_latent_dim", config.vae_latent_dim
        )

        if generator_type == "mae_vae":
            generative_model = mae_vae(
                vae_latent_dim=vae_latent_dim,
                freeze_encoder=True,
            ).cuda()
            print(f"Loaded MAE-VAE with latent dimension: {vae_latent_dim}")
        elif generator_type == "mae":
            generative_model = standard_mae().cuda()
            print("Loaded standard MAE.")
        else:
            raise ValueError(
                f"Unsupported generator_type '{generator_type}'. "
                "RFFR-MVAE-Wavelet supports only 'mae' and 'mae_vae'."
            )

        generative_model.load_state_dict(ckpt["state_dict"], strict=False)
        print("Successfully loaded model weights from checkpoint")

        self.inpainter = generative_model
        self.magic = torch.Tensor([0, 1, 2, 14, 15, 16, 28, 29, 30])
        self.full_patches = self._block_to_patch(torch.arange(16).unsqueeze(0))

    def _block_to_patch(self, idx, test=False):
        bs, blk = idx.shape
        if test:
            return self.full_patches.repeat(bs, 1)

        w = 3 * torch.div(idx, 4, rounding_mode="floor") + 1
        h = 3 * (idx % 4) + 1
        idx = w * 14 + h

        idx = idx.repeat(1, 9).reshape(bs, -1, blk).transpose(1, 2)
        idx = idx + self.magic
        idx = idx.reshape(bs, -1)
        return idx

    def forward(self, rgb_01, test=False):
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
                else torch.LongTensor(blocks).unsqueeze(0).repeat(rgb_01.shape[0], 1)
            )

        with torch.no_grad():
            if config.wavelet_residual_branch:
                spatial_res, reconstructed_img, block_id = self.inpainter.patch_by_patch_DIFF(
                    rgb_01,
                    block_id=block_id,
                    test=test,
                    return_reconstructed=True,
                )
                original_wavelets = rgb_to_wavelet(rgb_01)
                reconstructed_wavelets = rgb_to_wavelet(reconstructed_img)
                res_outputs = {
                    "spatial_residuals": spatial_res,
                    "wavelet_residuals": original_wavelets - reconstructed_wavelets,
                }
            else:
                res_outputs, block_id = self.inpainter.patch_by_patch_DIFF(
                    rgb_01, block_id=block_id, test=test
                )

        patch_id = self._block_to_patch(block_id, test=test)
        num_blocks = block_id.shape[1] if block_id is not None else 16
        return res_outputs, patch_id, num_blocks


class DeepfakeDetector(nn.Module):
    """2-branch RFFR detector with optional 12-channel wavelet residual branch."""

    def __init__(self):
        super().__init__()

        self.backbone_1 = BlockClassifier(**VIT_CONFIG)
        self.backbone_2 = BlockClassifier(load_pretrained=True, **VIT_CONFIG)

        if config.wavelet_residual_branch:
            print(
                "Initializing 3-branch architecture "
                "(RGB + spatial residuals + wavelet residuals)"
            )
            vit_config_12ch = VIT_CONFIG.copy()
            vit_config_12ch["in_chans"] = 12
            self.backbone_3 = AdaptiveViT(**vit_config_12ch)
            if (
                config.use_imagenet_pretrain_for_wavelets
                and config.pretrained_weights
                and os.path.exists(config.pretrained_weights)
            ):
                self._load_pretrained_for_wavelet_residual_branch()
            else:
                print(
                    "Using random initialization for wavelet residual branch "
                    "(recommended for frequency domain)"
                )
            classifier_input_dim = VIT_CONFIG["embed_dim"] * 3
        else:
            print("Initializing 2-branch architecture (RGB + spatial residuals)")
            self.backbone_3 = None
            classifier_input_dim = VIT_CONFIG["embed_dim"] * 2

        self.classifier = nn.Linear(classifier_input_dim, 2)

    def _load_pretrained_for_wavelet_residual_branch(self):
        try:
            pretrained_state = torch.load(config.pretrained_weights)
            filtered_state = {
                k: v
                for k, v in pretrained_state.items()
                if not k.startswith("patch_embed") and not k.startswith("pos_embed")
            }
            self.backbone_3.load_state_dict(filtered_state, strict=False)
            print(
                "Loaded ImageNet pretrained weights for wavelet residual branch "
                "(excluding patch_embed)"
            )
        except Exception as e:
            print(f"Warning: Could not load pretrained weights for wavelet branch: {e}")

    def forward(self, res_data, rgb_norm, patch_id):
        rgb_features = self.backbone_1(rgb_norm, patch_id=patch_id)

        if config.wavelet_residual_branch:
            if not (
                isinstance(res_data, dict)
                and "spatial_residuals" in res_data
                and "wavelet_residuals" in res_data
            ):
                raise ValueError(
                    "Expected res_data dict with 'spatial_residuals' and "
                    "'wavelet_residuals' keys for 3-branch mode"
                )

            spatial_features = self.backbone_2(
                res_data["spatial_residuals"], patch_id=patch_id
            )
            wavelet_features = self.backbone_3(
                res_data["wavelet_residuals"], patch_id=patch_id
            )
            feature = torch.cat(
                [rgb_features, spatial_features, wavelet_features], dim=1
            )
        else:
            residual_features = self.backbone_2(res_data, patch_id=patch_id)
            feature = torch.cat([rgb_features, residual_features], dim=1)

        output = self.classifier(feature)
        return feature, output


class RFFRL(nn.Module):
    """RFFR pipeline: residual generator plus 2/3-branch detector."""

    def __init__(self):
        super().__init__()
        self.rg = ResidualGenerator()
        self.dd = DeepfakeDetector()

    def forward(self, rgb_01, rgb_norm, test=False, current_iter=None):
        res, patch_id, num_blocks = self.rg(rgb_01, test=test)
        feature, output = self.dd(res, rgb_norm, patch_id)
        return feature, output, num_blocks


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    dummy_01 = torch.rand(2, 3, 224, 224).cuda()
    dummy_norm = torch.randn(2, 3, 224, 224).cuda()
    net = RFFRL().cuda()

    result = net(dummy_01, dummy_norm, test=False)
    print(f"Feature shape: {result[0].shape}")
    print(f"Output shape: {result[1].shape}")
