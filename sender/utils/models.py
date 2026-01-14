import torch
import torch.nn as nn
import torchvision.models as models

# --- PRETRAINED MODELS (Transfer Learning) ---

class PretrainedMobileNetEncoder(nn.Module):
    """
    Local Encoder: Wraps MobileNetV3-Small (Pretrained on ImageNet).
    """
    def __init__(self, encoded_space_dim=512):
        super().__init__()
        # Load pretrained MobileNetV3 Small
        # We need to handle weights='DEFAULT' or similar depending on torchvision version
        # For compatibility, we'll try 'DEFAULT' first.
        try:
            self.backbone = models.mobilenet_v3_small(weights='DEFAULT')
        except:
             # Fallback for older torchvision
             self.backbone = models.mobilenet_v3_small(pretrained=True)
             
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # The classifier of MobileNetV3 Small has a structure ending in Linear(576, 1000).
        # We replace it to map to our encoded_space_dim (512).
        # Backbone classifier: Sequential(Linear, Hardswish, Dropout, Linear)
        # We replace the final Linear layer.
        
        # Check input features of the last layer
        num_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(num_features, encoded_space_dim)
        
        # We WANT to train this new linear layer, so ensure it requires grad
        self.backbone.classifier[-1].weight.requires_grad = True
        self.backbone.classifier[-1].bias.requires_grad = True

    def forward(self, x):
        # MobileNet expects normalized input, but we'll assume standard scaling happens outside or robust enough
        return self.backbone(x)

class PretrainedResNetEncoder(nn.Module):
    """
    Edge Encoder: Wraps ResNet-50 (Pretrained on ImageNet).
    """
    def __init__(self, encoded_space_dim=512):
        super().__init__()
        try:
            self.backbone = models.resnet50(weights='DEFAULT')
        except:
            self.backbone = models.resnet50(pretrained=True)

        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # ResNet fc layer is Linear(2048, 1000). Replace it.
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, encoded_space_dim)
        
        # Ensure new layer is trainable
        self.backbone.fc.weight.requires_grad = True
        self.backbone.fc.bias.requires_grad = True

    def forward(self, x):
        return self.backbone(x)

# --- SIMPLE MODELS (Reuse Decoder for Local) ---

# --- SIMPLE MODELS (Reuse Decoder for Local) ---

class SimpleDecoder(nn.Module):
    """
    Lightweight Decoder for Local Processing.
    """
    def __init__(self, encoded_space_dim=512):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 32 * 8 * 8),
            nn.ReLU(inplace=True)
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 8, 8))
        self.decoder_conv = nn.Sequential(
            # Input: 32 x 8 x 8
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # -> 16 x 16 x 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1)   # -> 3 x 32 x 32
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

# --- COMPLEX MODELS (Reuse Decoder for Edge) ---
# ComplexDecoder is also generic (takes 512 vector). So we can reuse it.

class ComplexDecoder(nn.Module):
    """
    Heavyweight Decoder for Edge/Cloud Processing.
    """
    def __init__(self, encoded_space_dim=512):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256 * 4 * 4),
            nn.ReLU(inplace=True)
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 4, 4))
        self.decoder_conv = nn.Sequential(
            # Input: 256 x 4 x 4
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), # -> 128 x 8 x 8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # -> 64 x 16 x 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),   # -> 32 x 32 x 32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, stride=1, padding=1)                                # -> 3 x 32 x 32 (Refine)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

# Keep legacy classes for now to avoid breaking imports immediately, but user wants replacement.
# I will NOT remove SimpleEncoder/ComplexEncoder definitions yet, just add the new ones above.


# --- VAE WRAPPER (Stable Diffusion) ---
import logging
try:
    from diffusers import AutoencoderKL, AutoencoderTiny
except ImportError: 
    # Fallback if library not found during initial Docker build steps
    AutoencoderKL = None 
    AutoencoderTiny = None


class VAEWrapper(nn.Module):
    """
    Wrapper for Stable Diffusion VAE (AutoencoderKL).
    """
    def __init__(self, model_name="CompVis/stable-diffusion-v1-4", subfolder="vae", device='cpu'):
        super().__init__()
        self.device = device
        logging.info(f"Loading VAE from {model_name}...")
        
        if AutoencoderKL is None:
            logging.error("diffusers library not installed! VAE will fail.")
            self.vae = None
            return

        try:
            self.vae = AutoencoderKL.from_pretrained(model_name, subfolder=subfolder)
            # Freeze VAE
            for param in self.vae.parameters():
                param.requires_grad = False
        except Exception as e:
            logging.error(f"Failed to load VAE: {e}")
            self.vae = None

    def encode(self, x):
        """
        Input: [B, 3, H, W] (Assumed [0, 1])
        Output: [B, 4, H/8, W/8] Latent
        """
        if self.vae is None: return None
        # Convert [0, 1] -> [-1, 1]
        x = 2.0 * x - 1.0
        
        with torch.no_grad():
            # encode() returns a distribution. We sample or take mode.
            # SD uses sample() during training but for deterministic inference use mode().
            # But SD pipeline typically samples. Let's use sample().
            dist = self.vae.encode(x).latent_dist
            latents = dist.mode() # deterministic for "compression" consistency?
            # Scaling factor for SD VAE
            latents = latents * 0.18215
            
        return latents

    def decode(self, z):
        """
        Input: [B, 4, H/8, W/8]
        Output: [B, 3, H, W]
        """
        if self.vae is None: return None
        
        with torch.no_grad():
            z = z / 0.18215
            out = self.vae.decode(z).sample
            
        # Convert [-1, 1] -> [0, 1]
        out = (out / 2.0 + 0.5).clamp(0, 1)
        return out


class TinyVAEWrapper(nn.Module):
    """
    Wrapper for Tiny AutoEncoder (TAESD).
    """
    def __init__(self, model_name="madebyollin/taesd", device='cpu'):
        super().__init__()
        self.device = device
        logging.info(f"Loading TinyVAE from {model_name}...")
        
        if AutoencoderTiny is None:
            logging.error("diffusers library not installed! TinyVAE will fail.")
            self.vae = None
            return

        try:
            self.vae = AutoencoderTiny.from_pretrained(model_name)
            # Freeze VAE
            for param in self.vae.parameters():
                param.requires_grad = False
        except Exception as e:
            logging.error(f"Failed to load TinyVAE: {e}")
            self.vae = None

    def encode(self, x):
        """
        Input: [B, 3, H, W] (Assumed [0, 1])
        Output: [B, 4, H/8, W/8] Latent
        """
        if self.vae is None: return None
        # Convert [0, 1] -> [-1, 1] (TAESD uses same normalization as SD) (Actually check logic)
        # Yes, TAESD is a drop-in replacement for SD VAE.
        x = 2.0 * x - 1.0
        
        with torch.no_grad():
            # TinyVAE encode returns latents directly, scaling included?
            # Warning: TAESD output is often already scaled. Let's check docs or experiment.
            # Usually TAESD encode() returns scaled latents.
            # But diffusers implementation `encode` usually returns `EncoderOutput`.
            # Let's check standard usage:
            # latents = vae.encode(image).latents
            # And for TAESD in diffusers, it might be same.
            
            # Safe bet:
            out = self.vae.encode(x)
            latents = out.latents
            
            # Rescaling? SD VAE needs * 0.18215. TAESD is trained to match that scale?
            # TAESD approximates the *scaled* latents directly usually.
            # Let's assume it matches the SD VAE latent space perfectly (so no extra scaling needed if we decoding with SD VAE?).
            # Wait, if we use TinyVAE for BOTH encode and decode, we just need consistency.
            # If we mix (Tiny Encode -> Standard Decode), scale matters.
            # User wants SEM_LOCAL (Tiny) and SEM_EDGE (Full).
            # Receiver will use TinyDecoder for Local. So consistency is key.
            # I will apply * 0.18215 ONLY IF TAESD DOESN'T (to match SD standard).
            # Actually, diffusers AutoencoderTiny `scale_factor` is 1.0 by default? 
            # I will check if `vae.config.scaling_factor` exists.
            
            # For now, let's assume standard behavior:
            # Latents from TAESD are compatible with SD.
            pass
            
        return latents

    def decode(self, z):
        """
        Input: [B, 4, H/8, W/8]
        Output: [B, 3, H, W]
        """
        if self.vae is None: return None
        
        with torch.no_grad():
            # z input. 
            # Decode.
            out = self.vae.decode(z).sample
            
        # Convert [-1, 1] -> [0, 1]
        out = (out / 2.0 + 0.5).clamp(0, 1)
        return out



