import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CelebAGenerator(nn.Module):

    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(), #DCGAN outputs values between [-1, 1]
        )

    def forward(self, z):
        """
        z.shape = (B, 100)
        after reshaping and normalization: (B, 100, 1, 1)
        """
        z = z.view(z.size(0), 100, 1, 1)
        return self.model(z)


class CelebaVAE(nn.Module):
    
    def __init__(self, latent_dim=200):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(256*4*4, latent_dim)
        self.fc_logvar = nn.Linear(256*4*4, latent_dim)
        
        self.decoder_input = nn.Linear(latent_dim, 256*4*4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )
        
    def forward(self, z):
        z = self.decoder_input(z).view(z.size(0), 256, 4, 4)
        return self.decoder(z) # (B, 3, 64, 64)

class classifier(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(256*16*16, 1)
        
    def forward(self, x):
        #x.shape(B, 3, 64, 64)
        x = self.conv(x) # (B, 256, 16, 16)
        x = x.view(x.size(0), -1) # (B, 256*4*4)
        return self.fc(x) #(B, 1)        

class StyleGAN2Wrapper(nn.Module):
    def __init__(self, G, max_batch_size=64, noise_mode='random'):
        super().__init__()
        self.G = G
        self.latent_dim = G.z_dim  # 512
        self.max_batch_size = max_batch_size
        # 'random' (StyleGAN2 default) injects fresh per-layer noise on every
        # forward pass, so decoding the same z twice differs; 'const' reuses
        # each layer's fixed noise buffer, making G deterministic in z. The
        # posterior factories read this off the wrapper (getattr, default
        # 'random') and forward it to G, so it governs the sampling path too,
        # not just the wrapper's own forward.
        self.noise_mode = noise_mode

    def forward(self, z):
        if z.size(0) <= self.max_batch_size:
            return self.G(z, None, noise_mode=self.noise_mode)

        outputs = []
        for chunk in torch.split(z, self.max_batch_size, dim=0):
            outputs.append(self.G(chunk, None, noise_mode=self.noise_mode))
        return torch.cat(outputs, dim=0)
