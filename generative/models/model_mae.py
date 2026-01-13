# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import numpy as np

from timm.models.vision_transformer import PatchEmbed, Block
from configs.config import config


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)

        out_chans = in_chans

        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * out_chans, bias=True
        )

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        self.magic = torch.Tensor([0, 1, 2, 14, 15, 16, 28, 29, 30])
        self.batch = (
            torch.arange(config.batch_size, dtype=torch.long)[:, None]
            .expand(config.batch_size, 9)
            .contiguous()
            .view(-1)
        )

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        img_channels = imgs.shape[1]
        x = imgs.reshape(shape=(imgs.shape[0], img_channels, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * img_channels))
        return x

    def unpatchify(self, x):
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        img_channels = x.shape[2] // (p * p)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, img_channels))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], img_channels, h * p, h * p))
        return imgs

    def merge_output(self, pred, mask, data):
        mask = mask.unsqueeze(2)
        pred_img = self.unpatchify(pred)

        masked_patchified_data = self.patchify(data) * (1 - mask)
        pred_patchified = self.patchify(pred_img)
        merge = self.unpatchify(masked_patchified_data + pred_patchified * mask)
        return merge

    def random_masking(self, x, mask_ratio, block=False, block_id=None):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)

        if block:
            if block_id is None:
                remove_row = torch.randint(10, [N, 1]) + 1
                remove_col = torch.randint(10, [N, 1]) + 1
            else:
                remove_row = torch.LongTensor([3 * (block_id // 4) + 1] * N).unsqueeze(
                    1
                )
                remove_col = torch.LongTensor([3 * (block_id % 4) + 1] * N).unsqueeze(1)

            block_remove = (remove_row * 14 + remove_col).expand(N, 9) + self.magic
            block_remove = block_remove.long()

            if N == config.batch_size:
                batch = self.batch
            else:
                batch = (
                    torch.arange(N, dtype=torch.long)[:, None]
                    .expand(N, 9)
                    .contiguous()
                    .view(-1)
                )
            block_remove = block_remove.view(-1)
            noise[batch, block_remove] += 1
            len_keep = 187

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        if block:
            return x_masked, mask, ids_restore, remove_row, remove_col

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, block=False, block_id=None):
        x = self.patch_embed(x)

        x = x + self.pos_embed[:, 1:, :]

        if block:
            x, mask, ids_restore, row, col = self.random_masking(
                x, mask_ratio, block, block_id
            )
        else:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if block:
            return x, mask, ids_restore, row, col

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)

        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum(dim=1) / mask.sum(dim=1)
        return loss

    def patch_by_patch_DIFF(
        self,
        data,
        block_id=None,
        show=False,
        test=False,
        return_reconstructed=False,
        return_wavelet_residuals=False,
    ):
        """
        Compute patch-by-patch differences between original and reconstructed images.
        
        Args:
            data: Input images [B, C, H, W]
            block_id: Block indices to reconstruct [B, num_blocks]
            show: If True, return reconstructed image for visualization
            return_reconstructed: If True, return both residual and reconstructed image
            return_wavelet_residuals: Deprecated parameter, kept for backward compatibility
        
        Returns:
            Residual between original and reconstructed images, optionally with reconstructed image
        """
        bs = data.shape[0]
        num_block = block_id.shape[1]

        pred_img = data.clone()

        block_id = block_id.cpu()

        for i in range(num_block):
            w = 3 * torch.div(block_id[:, i], 4, rounding_mode="floor") + 1
            h = 3 * ((block_id[:, i]) % 4) + 1
            _, pred, _ = self(data.float(), block=True, block_id=block_id[:, i])
            pred = self.unpatchify(pred)

            for j in range(bs):
                pred_patch = pred[
                    j, :, w[j] * 16 : (w[j] + 3) * 16, h[j] * 16 : (h[j] + 3) * 16
                ]
                pred_img[
                    j, :, w[j] * 16 : (w[j] + 3) * 16, h[j] * 16 : (h[j] + 3) * 16
                ] = pred_patch

        res = (pred_img - data) * 4
        res = torch.clamp(res, -1, 1)

        if show:
            return res, pred_img
        if return_reconstructed:
            return res, pred_img, block_id
        return res, block_id

    def forward(self, imgs, mask_ratio=0.75, block=False, block_id=None):
        if block:
            latent, mask, ids_restore, row, col = self.forward_encoder(
                imgs, mask_ratio, block, block_id
            )
        else:
            latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def mae_vit_base_patch16_symmetric(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=768,
        decoder_depth=12,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b
