import torch
import torch.nn as nn
from models.model_mae import MaskedAutoencoderViT, get_2d_sincos_pos_embed
from configs.config import config


class VariationalBottleneck(nn.Module):
    """Simple VAE bottleneck with single linear layers (original architecture)"""

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, input_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.xavier_uniform_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)
        nn.init.xavier_uniform_(self.fc_decode.weight)
        nn.init.zeros_(self.fc_decode.bias)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, deterministic=False):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        if deterministic:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)

        x_decoded = self.fc_decode(z)

        return x_decoded, mu, logvar


class SophisticatedVariationalBottleneck(nn.Module):
    """Sophisticated VAE bottleneck with multi-layer MLP, residual connections, and LayerNorm"""

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Calculate intermediate dimension
        intermediate_dim = (input_dim + latent_dim) // 2

        # Encoder pathway with residual connections
        self.encoder_fc1 = nn.Linear(input_dim, intermediate_dim)
        self.encoder_ln1 = nn.LayerNorm(intermediate_dim)
        self.encoder_fc2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.encoder_ln2 = nn.LayerNorm(intermediate_dim)
        
        self.fc_mu = nn.Linear(intermediate_dim, latent_dim)
        self.fc_logvar = nn.Linear(intermediate_dim, latent_dim)

        # Decoder pathway with residual connections
        self.decoder_fc1 = nn.Linear(latent_dim, intermediate_dim)
        self.decoder_ln1 = nn.LayerNorm(intermediate_dim)
        self.decoder_fc2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.decoder_ln2 = nn.LayerNorm(intermediate_dim)
        self.decoder_fc3 = nn.Linear(intermediate_dim, input_dim)

        # Residual projection layers
        self.encoder_residual_proj = nn.Linear(input_dim, intermediate_dim)
        self.decoder_residual_proj = nn.Linear(latent_dim, intermediate_dim)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        # Initialize all linear layers with xavier uniform
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        # First encoder block with residual
        residual = self.encoder_residual_proj(x)
        h = self.encoder_fc1(x)
        h = self.encoder_ln1(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = h + residual  # Residual connection

        # Second encoder block with residual
        residual = h
        h = self.encoder_fc2(h)
        h = self.encoder_ln2(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = h + residual  # Residual connection

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar

    def decode(self, z):
        # First decoder block with residual
        residual = self.decoder_residual_proj(z)
        h = self.decoder_fc1(z)
        h = self.decoder_ln1(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = h + residual  # Residual connection

        # Second decoder block with residual
        residual = h
        h = self.decoder_fc2(h)
        h = self.decoder_ln2(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = h + residual  # Residual connection

        # Final projection
        x_decoded = self.decoder_fc3(h)
        
        return x_decoded

    def forward(self, x, deterministic=False):
        mu, logvar = self.encode(x)

        if deterministic:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)

        x_decoded = self.decode(z)

        return x_decoded, mu, logvar


class HybridMAEVAE(MaskedAutoencoderViT):

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
        vae_latent_dim=512,
        freeze_encoder=True,
        vae_bottleneck_type="simple",
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            norm_pix_loss=norm_pix_loss,
        )

        self.vae_latent_dim = vae_latent_dim
        self.freeze_encoder = freeze_encoder
        self.vae_bottleneck_type = vae_bottleneck_type

        # Select bottleneck architecture based on config
        if vae_bottleneck_type == "sophisticated":
            print(f"Using SOPHISTICATED VAE bottleneck (input_dim={embed_dim}, latent_dim={vae_latent_dim})")
            self.vae_bottleneck = SophisticatedVariationalBottleneck(
                input_dim=embed_dim, latent_dim=vae_latent_dim
            )
        else:
            print(f"Using SIMPLE VAE bottleneck (input_dim={embed_dim}, latent_dim={vae_latent_dim})")
            self.vae_bottleneck = VariationalBottleneck(
                input_dim=embed_dim, latent_dim=vae_latent_dim
            )

        if freeze_encoder:
            self.freeze_mae_encoder()

    def freeze_mae_encoder(self):
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        for param in self.blocks.parameters():
            param.requires_grad = False
        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = False
        self.norm.requires_grad_(False)

    def unfreeze_mae_encoder(self):
        for param in self.patch_embed.parameters():
            param.requires_grad = True
        for param in self.blocks.parameters():
            param.requires_grad = True
        self.cls_token.requires_grad = True
        self.norm.requires_grad_(True)

    def forward_encoder_with_vae(
        self, x, mask_ratio, block=False, block_id=None, deterministic=False
    ):
        encoder_result = self.forward_encoder(x, mask_ratio, block, block_id)

        if len(encoder_result) == 5:
            latent, mask, ids_restore, row, col = encoder_result
        else:
            latent, mask, ids_restore = encoder_result
            row, col = None, None

        vae_latent, mu, logvar = self.vae_bottleneck(
            latent, deterministic=deterministic
        )

        if block and row is not None and col is not None:
            return vae_latent, mask, ids_restore, mu, logvar, row, col
        else:
            return vae_latent, mask, ids_restore, mu, logvar

    def forward(
        self,
        imgs,
        mask_ratio=0.75,
        block=False,
        block_id=None,
        deterministic=False,
        beta=0.01,
    ):
        encoder_outputs = self.forward_encoder_with_vae(
            imgs, mask_ratio, block, block_id, deterministic
        )

        if len(encoder_outputs) == 7:
            latent, mask, ids_restore, mu, logvar, row, col = encoder_outputs
        else:
            latent, mask, ids_restore, mu, logvar = encoder_outputs

        pred = self.forward_decoder(latent, ids_restore)
        loss_recon = self.forward_loss(imgs, pred, mask)

        kl_loss = self.compute_kl_loss(mu, logvar)

        loss_total = loss_recon.mean() + beta * kl_loss

        return loss_total.mean() if loss_total.dim() > 0 else loss_total, pred, mask, loss_recon.mean(), kl_loss.mean() if kl_loss.dim() > 0 else kl_loss

    def compute_kl_loss(self, mu, logvar):
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / mu.size(0)
        return kl_loss

    def patch_by_patch_DIFF(
        self,
        data,
        block_id=None,
        show=False,
        test=False,
        return_reconstructed=False,
        return_wavelet_residuals=False,  # Deprecated, kept for backward compatibility
        return_wavelet_dual=False,  # Deprecated, kept for backward compatibility
    ):
        """
        Compute patch-by-patch differences for MAE-VAE model.
        
        Args:
            data: Input images
            block_id: Block indices to reconstruct
            show: Return reconstructed image for visualization
            test: If True, do not mask any patches
            return_reconstructed: Return both residual and reconstructed image
            return_wavelet_residuals: Deprecated parameter
            return_wavelet_dual: Deprecated parameter
        
        Returns:
            Residual between original and reconstructed images
        """
        if test:
            mask_ratio = 0
        else:
            mask_ratio = 0.75

        bs = data.shape[0]
        if test or block_id is None:
            block_id = torch.arange(16).unsqueeze(0).repeat(bs, 1)
        num_block = block_id.shape[1]

        pred_img = data.clone()
        block_id = block_id.cpu()

        for i in range(num_block):
            w = 3 * torch.div(block_id[:, i], 4, rounding_mode="floor") + 1
            h = 3 * ((block_id[:, i]) % 4) + 1

            with torch.no_grad():
                encoder_outputs = self.forward_encoder_with_vae(
                    data.float(),
                    mask_ratio,
                    block=True,
                    block_id=block_id[:, i],
                    deterministic=True,
                )
                if len(encoder_outputs) == 7:
                    latent, mask, ids_restore, mu, logvar, row, col = encoder_outputs
                else:
                    latent, mask, ids_restore, mu, logvar = encoder_outputs

                pred = self.forward_decoder(latent, ids_restore)

            pred = self.unpatchify(pred)

            for j in range(bs):
                pred_patch = pred[
                    j, :, w[j] * 16 : (w[j] + 3) * 16, h[j] * 16 : (h[j] + 3) * 16
                ]
                pred_img[
                    j, :, w[j] * 16 : (w[j] + 3) * 16, h[j] * 16 : (h[j] + 3) * 16
                ] = pred_patch

        res = data - pred_img
        
        if show:
            return res, pred_img
        if return_reconstructed:
            return res, pred_img, block_id
        return res, block_id


def mae_vae_vit_base_patch16(**kwargs):
    model = HybridMAEVAE(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=nn.LayerNorm,
        **kwargs,
    )
    return model


if __name__ == "__main__":
    model = mae_vae_vit_base_patch16(vae_latent_dim=512, freeze_encoder=True, vae_bottleneck_type="sophisticated")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    dummy_input = torch.randn(2, 3, 224, 224)
    loss_total, pred, mask = model(dummy_input, mask_ratio=0.75, beta=0.001)
    print(f"Total loss: {loss_total.item():.4f}")
    print(f"Prediction shape: {pred.shape}")
