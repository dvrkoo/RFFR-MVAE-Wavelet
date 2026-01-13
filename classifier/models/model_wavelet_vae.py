import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wavelet_utils import (
    pytorch_dwt2,
    pytorch_idwt2,
    PYWAVELETS_AVAILABLE,
    PYTORCH_WAVELETS_AVAILABLE,
)


class WaveletEncoder(nn.Module):

    def __init__(self, latent_dim=256, in_chans=12, base_channels=64):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(
            in_chans, base_channels, kernel_size=4, stride=2, padding=1
        )
        self.bn1 = nn.BatchNorm2d(base_channels)

        self.conv2 = nn.Conv2d(
            base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1
        )
        self.bn2 = nn.BatchNorm2d(base_channels * 2)

        self.conv3 = nn.Conv2d(
            base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1
        )
        self.bn3 = nn.BatchNorm2d(base_channels * 4)

        self.conv4 = nn.Conv2d(
            base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1
        )
        self.bn4 = nn.BatchNorm2d(base_channels * 8)

        self.conv5 = nn.Conv2d(
            base_channels * 8, base_channels * 8, kernel_size=4, stride=2, padding=1
        )
        self.bn5 = nn.BatchNorm2d(base_channels * 8)

        self.flatten_dim = base_channels * 8 * 7 * 7

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class WaveletDecoder(nn.Module):

    def __init__(self, latent_dim=256, out_chans=12, base_channels=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.out_chans = out_chans

        self.fc = nn.Linear(latent_dim, base_channels * 8 * 7 * 7)

        self.deconv1 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 8, kernel_size=4, stride=2, padding=1
        )
        self.bn1 = nn.BatchNorm2d(base_channels * 8)

        self.deconv2 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1
        )
        self.bn2 = nn.BatchNorm2d(base_channels * 4)

        self.deconv3 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1
        )
        self.bn3 = nn.BatchNorm2d(base_channels * 2)

        self.deconv4 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1
        )
        self.bn4 = nn.BatchNorm2d(base_channels)

        self.deconv5 = nn.ConvTranspose2d(
            base_channels, out_chans, kernel_size=4, stride=2, padding=1
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), self.base_channels * 8, 7, 7)

        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        x = self.deconv5(x)

        return x


class WaveletVAE(nn.Module):

    def __init__(
        self,
        latent_dim=256,
        in_chans=3,
        base_channels=64,
        wavelet_type="db4",
        norm_pix_loss=False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_chans = in_chans
        self.wavelet_type = wavelet_type
        self.norm_pix_loss = norm_pix_loss

        if not (PYWAVELETS_AVAILABLE or PYTORCH_WAVELETS_AVAILABLE):
            raise RuntimeError(
                "PyWavelets or pytorch-wavelets is required for WaveletVAE but neither is available"
            )

        self.encoder = WaveletEncoder(
            latent_dim=latent_dim, in_chans=in_chans * 4, base_channels=base_channels
        )
        self.decoder = WaveletDecoder(
            latent_dim=latent_dim, out_chans=in_chans * 4, base_channels=base_channels
        )

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def rgb_to_wavelet_concat(self, imgs):
        LL, LH, HL, HH = pytorch_dwt2(imgs, self.wavelet_type)

        H_orig, W_orig = imgs.shape[2], imgs.shape[3]
        H_wavelet, W_wavelet = LL.shape[2], LL.shape[3]

        if (H_wavelet, W_wavelet) != (H_orig, W_orig):
            LL = F.interpolate(
                LL, size=(H_orig, W_orig), mode="bilinear", align_corners=False
            )
            LH = F.interpolate(
                LH, size=(H_orig, W_orig), mode="bilinear", align_corners=False
            )
            HL = F.interpolate(
                HL, size=(H_orig, W_orig), mode="bilinear", align_corners=False
            )
            HH = F.interpolate(
                HH, size=(H_orig, W_orig), mode="bilinear", align_corners=False
            )

        wavelet_concat = torch.cat([LL, LH, HL, HH], dim=1)
        return wavelet_concat, (LL, LH, HL, HH)

    def wavelet_concat_to_rgb(self, wavelet_concat):
        B, C, H, W = wavelet_concat.shape
        channels_per_band = C // 4

        LL = wavelet_concat[:, :channels_per_band, :, :]
        LH = wavelet_concat[:, channels_per_band : 2 * channels_per_band, :, :]
        HL = wavelet_concat[:, 2 * channels_per_band : 3 * channels_per_band, :, :]
        HH = wavelet_concat[:, 3 * channels_per_band :, :, :]

        imgs_rec = pytorch_idwt2(LL, LH, HL, HH, self.wavelet_type)

        if imgs_rec.shape[2:] != (H, W):
            imgs_rec = F.interpolate(
                imgs_rec, size=(H, W), mode="bilinear", align_corners=False
            )

        return imgs_rec

    def forward(self, imgs, beta=0.01):
        wavelet_concat, (LL, LH, HL, HH) = self.rgb_to_wavelet_concat(imgs)

        mu, logvar = self.encode(wavelet_concat)
        z = self.reparameterize(mu, logvar)
        wavelet_rec = self.decode(z)

        reconstructed = self.wavelet_concat_to_rgb(wavelet_rec)

        recon_loss = F.mse_loss(reconstructed, imgs, reduction="mean")

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / imgs.size(0)

        total_loss = recon_loss + beta * kl_loss

        return total_loss, reconstructed, None

    def reconstruct(self, x, deterministic=True):
        with torch.no_grad():
            wavelet_concat, _ = self.rgb_to_wavelet_concat(x)
            mu, logvar = self.encode(wavelet_concat)
            if deterministic:
                z = mu
            else:
                z = self.reparameterize(mu, logvar)
            wavelet_rec = self.decode(z)
            reconstructed = self.wavelet_concat_to_rgb(wavelet_rec)
            residuals = x - reconstructed
        return residuals

    def get_reconstruction_and_residuals(self, x, deterministic=True):
        with torch.no_grad():
            wavelet_concat, _ = self.rgb_to_wavelet_concat(x)
            mu, logvar = self.encode(wavelet_concat)
            if deterministic:
                z = mu
            else:
                z = self.reparameterize(mu, logvar)
            wavelet_rec = self.decode(z)
            reconstructed = self.wavelet_concat_to_rgb(wavelet_rec)
            residuals = x - reconstructed
        return reconstructed, residuals


def wavelet_vae_base(
    latent_dim=256, in_chans=3, base_channels=64, wavelet_type="db4", **kwargs
):
    model = WaveletVAE(
        latent_dim=latent_dim,
        in_chans=in_chans,
        base_channels=base_channels,
        wavelet_type=wavelet_type,
        **kwargs,
    )
    return model


if __name__ == "__main__":
    if not (PYWAVELETS_AVAILABLE or PYTORCH_WAVELETS_AVAILABLE):
        print(
            "ERROR: PyWavelets or pytorch-wavelets required. Install with: pip install PyWavelets pytorch-wavelets"
        )
        sys.exit(1)

    model = wavelet_vae_base(latent_dim=256, base_channels=64)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    dummy_input = torch.randn(2, 3, 224, 224)
    total_loss, reconstructed, _ = model(dummy_input, beta=0.01)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Reconstruction shape: {reconstructed.shape}")

    residuals = model.reconstruct(dummy_input)
    print(f"Residuals shape: {residuals.shape}")
