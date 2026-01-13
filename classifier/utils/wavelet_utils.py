import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Try to import pytorch_wavelets, provide fallback if not available
try:
    from pytorch_wavelets import DWTForward, DWTInverse

    PYTORCH_WAVELETS_AVAILABLE = True
except Exception as e:
    print(f"pytorch_wavelets import failed: {e}")
    PYTORCH_WAVELETS_AVAILABLE = False
    DWTForward = None
    DWTInverse = None

# Try to import pywt for generative-style transforms
try:
    import pywt

    PYWAVELETS_AVAILABLE = True
    pywt_module = pywt
except Exception as e:
    PYWAVELETS_AVAILABLE = False
    pywt_module = None

from configs.config import config

_fallback_warning_printed = False

# GPU-based Wavelet Modules (initialized only if needed)
dwt_transform = None
idwt_transform = None


def _initialize_wavelet_transforms(tensor):
    """Initialize DWT and IDWT transforms on the correct device."""
    global dwt_transform, idwt_transform

    if not PYTORCH_WAVELETS_AVAILABLE or DWTForward is None or DWTInverse is None:
        return

    device = tensor.device

    # Check if initialization is needed
    needs_init = dwt_transform is None
    if not needs_init and dwt_transform is not None:
        try:
            transform_device = next(iter(dwt_transform.buffers())).device
            needs_init = transform_device != device
        except (StopIteration, AttributeError):
            needs_init = True

    if needs_init:
        print(f"Initializing wavelet transforms on device: {device}")
        dwt_transform = DWTForward(
            J=config.wavelet_levels, wave=config.wavelet_type, mode="symmetric"
        ).to(device)
        idwt_transform = DWTInverse(wave=config.wavelet_type, mode="symmetric").to(
            device
        )


def _fallback_frequency_analysis(tensor):
    """
    Fallback frequency analysis using Sobel filters for gradient-based high-frequency detection.
    This runs entirely on the GPU.
    """
    device = tensor.device

    # Sobel filters for edge detection (approximates high-frequency content)
    sobel_x = (
        torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    sobel_y = (
        torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )

    # Apply to each channel independently
    b, c, h, w = tensor.shape
    tensor_flat = tensor.view(b * c, 1, h, w)

    grad_x = F.conv2d(tensor_flat, sobel_x, padding=1)
    grad_y = F.conv2d(tensor_flat, sobel_y, padding=1)

    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    return gradient_magnitude.view(b, c, h, w)


def extract_all_wavelet_subbands(residuals):
    """
    Extract all 4 wavelet subbands at native resolution for the 4-branch architecture.

    Args:
        residuals: Spatial residuals tensor (B, C, H, W)

    Returns:
        dict: Dictionary with keys 'LL', 'LH', 'HL', 'HH' containing subbands
    """
    if not config.use_wavelets:
        # Return zeros at original resolution for all subbands
        B, C, H, W = residuals.shape
        device = residuals.device
        H_sub, W_sub = H // 2, W // 2
        return {
            "LL": torch.zeros(B, C, H_sub, W_sub, device=device),
            "LH": torch.zeros(B, C, H_sub, W_sub, device=device),
            "HL": torch.zeros(B, C, H_sub, W_sub, device=device),
            "HH": torch.zeros(B, C, H_sub, W_sub, device=device),
        }

    if not PYTORCH_WAVELETS_AVAILABLE:
        # Use gradient-based approximation as fallback
        print("Wavelet subbands not available, using gradient approximation")
        gradient_mag = _fallback_frequency_analysis(residuals)
        B, C, H, W = gradient_mag.shape
        H_sub, W_sub = H // 2, W // 2

        downsampled = F.interpolate(
            gradient_mag, size=(H_sub, W_sub), mode="bilinear", align_corners=False
        )

        return {
            "LL": downsampled * 0.5,  # Low-freq approximation
            "LH": downsampled * 0.8,  # Vertical edges
            "HL": downsampled * 0.8,  # Horizontal edges
            "HH": downsampled * 1.0,  # Diagonal edges (highest freq)
        }

    _initialize_wavelet_transforms(residuals)

    if dwt_transform is None:
        # Fallback if initialization failed
        gradient_mag = _fallback_frequency_analysis(residuals)
        B, C, H, W = gradient_mag.shape
        H_sub, W_sub = H // 2, W // 2
        downsampled = F.interpolate(
            gradient_mag, size=(H_sub, W_sub), mode="bilinear", align_corners=False
        )

        return {
            "LL": downsampled * 0.5,
            "LH": downsampled * 0.8,
            "HL": downsampled * 0.8,
            "HH": downsampled * 1.0,
        }

    try:
        # Decompose residuals to get wavelet coefficients
        Yl, Yh = dwt_transform(residuals)

        if len(Yh) == 0:
            # No high-frequency coefficients, return zeros
            B, C, H, W = residuals.shape
            H_sub, W_sub = H // 2, W // 2
            device = residuals.device
            return {
                "LL": torch.zeros(B, C, H_sub, W_sub, device=device),
                "LH": torch.zeros(B, C, H_sub, W_sub, device=device),
                "HL": torch.zeros(B, C, H_sub, W_sub, device=device),
                "HH": torch.zeros(B, C, H_sub, W_sub, device=device),
            }

        # Extract the finest level coefficients
        finest_level_coeffs = Yh[0]  # Shape: (B, C, 3, H_level, W_level)

        # LL is the approximation coefficients
        LL = Yl  # Shape: (B, C, H_level, W_level)

        # Extract the 3 detail bands: LH, HL, HH
        LH = finest_level_coeffs[:, :, 0, :, :]  # Vertical edges
        HL = finest_level_coeffs[:, :, 1, :, :]  # Horizontal edges
        HH = finest_level_coeffs[:, :, 2, :, :]  # Diagonal edges

        # Apply enhancement if configured
        enhancement_factor = config.wavelet_high_freq_weight
        LL = LL * 0.5  # Keep LL relatively unchanged
        LH = LH * enhancement_factor
        HL = HL * enhancement_factor
        HH = HH * enhancement_factor

        # Normalize to preserve relative importance
        subbands = {
            "LL": torch.clamp(LL, -1, 1),
            "LH": torch.clamp(LH, -1, 1),
            "HL": torch.clamp(HL, -1, 1),
            "HH": torch.clamp(HH, -1, 1),
        }

        return subbands

    except Exception as e:
        print(f"Wavelet subband extraction failed: {e}, using fallback")
        # Fallback: use gradient-based approximation
        gradient_mag = _fallback_frequency_analysis(residuals)
        B, C, H, W = gradient_mag.shape
        H_sub, W_sub = H // 2, W // 2
        downsampled = F.interpolate(
            gradient_mag, size=(H_sub, W_sub), mode="bilinear", align_corners=False
        )

        return {
            "LL": downsampled * 0.5,
            "LH": downsampled * 0.8,
            "HL": downsampled * 0.8,
            "HH": downsampled * 1.0,
        }


def create_fused_wavelet_representation(residuals):
    """
    Create a fused wavelet representation for 3-branch architecture without upsampling.
    Combines all 4 wavelet subbands (LL, LH, HL, HH) into a 3-channel representation
    while preserving native resolution to avoid artifacts.

    Args:
        residuals: Spatial residuals tensor (B, C, H, W)

    Returns:
        fused_wavelets: Tensor (B, 3, H_native, W_native) containing fused wavelet information
                       - Channel 0: LL (approximation)
                       - Channel 1: Combined LH+HL (horizontal+vertical details)
                       - Channel 2: HH (diagonal details)
    """
    if not config.use_wavelets:
        # Return zeros at original resolution if wavelets disabled
        return torch.zeros(
            residuals.shape[0],
            3,
            residuals.shape[2],
            residuals.shape[3],
            device=residuals.device,
        )

    if not PYTORCH_WAVELETS_AVAILABLE:
        print(
            "Wavelets not available, using gradient-based approximation for fused representation"
        )
        return _fallback_frequency_analysis(residuals).repeat(1, 3, 1, 1)

    _initialize_wavelet_transforms(residuals)

    if dwt_transform is None:
        # Fallback if initialization failed
        gradient_mag = _fallback_frequency_analysis(residuals)
        return (
            gradient_mag.repeat(1, 3, 1, 1)
            if gradient_mag.shape[1] == 1
            else gradient_mag[:, :3]
        )

    try:
        # Decompose residuals to get wavelet coefficients
        Yl, Yh = dwt_transform(residuals)

        if len(Yh) == 0:
            # No high-frequency coefficients, use approximation only
            B, C, H, W = Yl.shape
            LL_avg = Yl.mean(dim=1, keepdim=True)  # Average across channels
            zeros = torch.zeros_like(LL_avg)
            return torch.cat([LL_avg, zeros, zeros], dim=1)

        # Use the finest level (level 0) coefficients at native resolution
        finest_level_coeffs = Yh[0]  # Shape: (B, C, 3, H_level, W_level)
        B, C, _, H_level, W_level = finest_level_coeffs.shape

        # LL: Approximation coefficients (low-frequency)
        LL_channel = Yl.mean(dim=1, keepdim=True)  # (B, 1, H_LL, W_LL)

        # Detail coefficients: LH, HL, HH from finest level
        LH = finest_level_coeffs[:, :, 0].mean(dim=1, keepdim=True)  # Vertical details
        HL = finest_level_coeffs[:, :, 1].mean(
            dim=1, keepdim=True
        )  # Horizontal details
        HH = finest_level_coeffs[:, :, 2].mean(dim=1, keepdim=True)  # Diagonal details

        # Combine LH and HL into single channel for efficiency
        LH_HL_combined = (LH + HL) / 2  # Average of horizontal and vertical

        # Handle size differences - use the largest common size
        target_H = min(LL_channel.shape[2], LH_HL_combined.shape[2], HH.shape[2])
        target_W = min(LL_channel.shape[3], LH_HL_combined.shape[3], HH.shape[3])

        # Crop to common size (no upsampling/downsampling)
        LL_cropped = LL_channel[:, :, :target_H, :target_W]
        LH_HL_cropped = LH_HL_combined[:, :, :target_H, :target_W]
        HH_cropped = HH[:, :, :target_H, :target_W]

        # Combine into 3-channel representation
        fused_wavelets = torch.cat([LL_cropped, LH_HL_cropped, HH_cropped], dim=1)

        # Apply enhancement if configured
        # Enhance high-frequency components more than approximation
        fused_wavelets[:, 0:1] *= 1.0  # Keep LL as is
        fused_wavelets[:, 1:3] *= config.wavelet_high_freq_weight  # Enhance details

        # Normalize to preserve relative importance
        fused_wavelets = torch.clamp(fused_wavelets, -1, 1)

        return fused_wavelets

    except Exception as e:
        print(f"Fused wavelet representation failed: {e}, using fallback")
        # Fallback: use gradient-based approximation
        gradient_mag = _fallback_frequency_analysis(residuals)
        return (
            gradient_mag.repeat(1, 3, 1, 1)
            if gradient_mag.shape[1] == 1
            else gradient_mag[:, :3]
        )


def _pywt_dwt2(imgs):
    if not PYWAVELETS_AVAILABLE or pywt_module is None:
        raise RuntimeError("PyWavelets is not available")

    device = imgs.device
    dtype = imgs.dtype
    B, C, H, W = imgs.shape

    imgs_np = imgs.detach().cpu().numpy()

    LL_list = []
    LH_list = []
    HL_list = []
    HH_list = []

    for b in range(B):
        for c in range(C):
            coeffs = pywt_module.dwt2(
                imgs_np[b, c], config.wavelet_type, mode="symmetric"
            )
            LL, (LH, HL, HH) = coeffs
            LL_list.append(LL)
            LH_list.append(LH)
            HL_list.append(HL)
            HH_list.append(HH)

    LL = np.stack(LL_list, axis=0).reshape(
        B, C, LL_list[0].shape[0], LL_list[0].shape[1]
    )
    LH = np.stack(LH_list, axis=0).reshape(
        B, C, LH_list[0].shape[0], LH_list[0].shape[1]
    )
    HL = np.stack(HL_list, axis=0).reshape(
        B, C, HL_list[0].shape[0], HL_list[0].shape[1]
    )
    HH = np.stack(HH_list, axis=0).reshape(
        B, C, HH_list[0].shape[0], HH_list[0].shape[1]
    )

    LL = torch.from_numpy(LL).to(device=device, dtype=dtype)
    LH = torch.from_numpy(LH).to(device=device, dtype=dtype)
    HL = torch.from_numpy(HL).to(device=device, dtype=dtype)
    HH = torch.from_numpy(HH).to(device=device, dtype=dtype)

    return LL, LH, HL, HH


def _pywt_idwt2(LL, LH, HL, HH):
    if not PYWAVELETS_AVAILABLE or pywt_module is None:
        raise RuntimeError("PyWavelets is not available")

    device = LL.device
    dtype = LL.dtype
    B, C, H, W = LL.shape

    LL_np = LL.detach().cpu().numpy()
    LH_np = LH.detach().cpu().numpy()
    HL_np = HL.detach().cpu().numpy()
    HH_np = HH.detach().cpu().numpy()

    reconstructed_list = []

    for b in range(B):
        for c in range(C):
            coeffs = (LL_np[b, c], (LH_np[b, c], HL_np[b, c], HH_np[b, c]))
            recon = pywt_module.idwt2(coeffs, config.wavelet_type, mode="symmetric")
            reconstructed_list.append(recon)

    reconstructed = np.stack(reconstructed_list, axis=0).reshape(
        B, C, reconstructed_list[0].shape[0], reconstructed_list[0].shape[1]
    )
    reconstructed = torch.from_numpy(reconstructed).to(device=device, dtype=dtype)

    return reconstructed


def rgb_to_wavelet(imgs):
    global _fallback_warning_printed

    if not config.classifier_uses_wavelets:
        return imgs

    if not PYWAVELETS_AVAILABLE:
        if not _fallback_warning_printed:
            print("PyWavelets not available, using gradient-based fallback")
            _fallback_warning_printed = True
        return _fallback_frequency_analysis(imgs)

    try:
        LL, LH, HL, HH = _pywt_dwt2(imgs)

        LH = LH * config.wavelet_high_freq_weight
        HL = HL * config.wavelet_high_freq_weight
        HH = HH * config.wavelet_high_freq_weight

        orig_h, orig_w = imgs.shape[2], imgs.shape[3]
        LL_upsampled = F.interpolate(
            LL, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
        LH_upsampled = F.interpolate(
            LH, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
        HL_upsampled = F.interpolate(
            HL, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
        HH_upsampled = F.interpolate(
            HH, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )

        fused = torch.cat(
            [LL_upsampled, LH_upsampled, HL_upsampled, HH_upsampled], dim=1
        )

        return fused

    except Exception as e:
        if not _fallback_warning_printed:
            print(f"Wavelet transform failed: {e}, using gradient-based fallback")
            _fallback_warning_printed = True
        return _fallback_frequency_analysis(imgs)


def wavelet_to_rgb(wavelet_coeffs):
    if not config.classifier_uses_wavelets:
        return wavelet_coeffs

    if not PYWAVELETS_AVAILABLE:
        return wavelet_coeffs

    try:
        B, C, H, W = wavelet_coeffs.shape

        if C == 3:
            return wavelet_coeffs

        if C == 12:
            num_channels = C // 4
            LL = wavelet_coeffs[:, :num_channels, :, :]
            LH = (
                wavelet_coeffs[:, num_channels : 2 * num_channels, :, :]
                / config.wavelet_high_freq_weight
            )
            HL = (
                wavelet_coeffs[:, 2 * num_channels : 3 * num_channels, :, :]
                / config.wavelet_high_freq_weight
            )
            HH = (
                wavelet_coeffs[:, 3 * num_channels :, :, :]
                / config.wavelet_high_freq_weight
            )
        elif C == 9:
            num_channels = C // 3
            LL = wavelet_coeffs[:, :num_channels, :, :]
            LH_HL = wavelet_coeffs[:, num_channels : 2 * num_channels, :, :]
            HH = wavelet_coeffs[:, 2 * num_channels :, :, :]
            LH = LH_HL / config.wavelet_high_freq_weight
            HL = LH_HL / config.wavelet_high_freq_weight
            HH = HH / config.wavelet_high_freq_weight
        else:
            return wavelet_coeffs

        orig_size = (H, W)

        device = wavelet_coeffs.device
        dtype = wavelet_coeffs.dtype

        LL_np = LL.detach().cpu().numpy()
        LH_np = LH.detach().cpu().numpy()
        HL_np = HL.detach().cpu().numpy()
        HH_np = HH.detach().cpu().numpy()

        reconstructed_list = []

        for b in range(B):
            for c in range(num_channels):
                test_coeffs = pywt_module.dwt2(
                    np.zeros((H, W)), config.wavelet_type, mode="symmetric"
                )
                target_LL, _ = test_coeffs
                wavelet_h, wavelet_w = target_LL.shape

                LL_single = (
                    F.interpolate(
                        LL[b : b + 1, c : c + 1],
                        size=(wavelet_h, wavelet_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze()
                    .detach()
                    .cpu()
                    .numpy()
                )

                LH_single = (
                    F.interpolate(
                        LH[b : b + 1, c : c + 1],
                        size=(wavelet_h, wavelet_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze()
                    .detach()
                    .cpu()
                    .numpy()
                )

                HL_single = (
                    F.interpolate(
                        HL[b : b + 1, c : c + 1],
                        size=(wavelet_h, wavelet_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze()
                    .detach()
                    .cpu()
                    .numpy()
                )

                HH_single = (
                    F.interpolate(
                        HH[b : b + 1, c : c + 1],
                        size=(wavelet_h, wavelet_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze()
                    .detach()
                    .cpu()
                    .numpy()
                )

                coeffs = (LL_single, (LH_single, HL_single, HH_single))
                recon = pywt_module.idwt2(coeffs, config.wavelet_type, mode="symmetric")

                if recon.shape != (H, W):
                    recon_tensor = torch.from_numpy(recon).unsqueeze(0).unsqueeze(0)
                    recon_tensor = F.interpolate(
                        recon_tensor, size=(H, W), mode="bilinear", align_corners=False
                    )
                    recon = recon_tensor.squeeze().numpy()

                reconstructed_list.append(recon)

        reconstructed = np.stack(reconstructed_list, axis=0).reshape(
            B, num_channels, H, W
        )
        reconstructed = torch.from_numpy(reconstructed).to(device=device, dtype=dtype)

        return reconstructed

    except Exception as e:
        print(f"Wavelet inverse failed: {e}, returning coeffs")
        import traceback

        traceback.print_exc()
        return wavelet_coeffs
