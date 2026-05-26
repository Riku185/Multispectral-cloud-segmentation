
import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetUNet(nn.Module):

    def __init__(self, input_channels=4):
        super().__init__()

        backbone = models.mobilenet_v2(pretrained=True)

        self.encoder = backbone.features

        self.first_conv = nn.Conv2d(input_channels, 3, kernel_size=1)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280,256,2,2),
            nn.ReLU(),
            nn.ConvTranspose2d(256,128,2,2),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,2,2),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,2,2),
            nn.ReLU(),
            nn.Conv2d(32,3,1)
        )

    def forward(self,x):

        x = self.first_conv(x)
        x = self.encoder(x)
        x = self.decoder(x)

        return x
