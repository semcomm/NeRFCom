import torch.nn as nn

from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN

class NTC_model(nn.Module):
    def __init__(self, N=32):
        super().__init__()

        self.entropy_bottleneck = EntropyBottleneck(N)

        self.encode = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32, N, stride=2, kernel_size=5, padding=2),
                GDN(N)
            ),
            nn.Sequential(
                nn.Conv2d(N, N, stride=2, kernel_size=5, padding=2),
                GDN(N)
            ),
            nn.Sequential(
                nn.Conv2d(N, N, stride=2, kernel_size=5, padding=2),
                GDN(N)
            )
        ])

        self.decode = nn.Sequential(
            nn.Sequential(
                nn.ConvTranspose2d(N, N, kernel_size=5, padding=2, output_padding=1, stride=2),
                GDN(N, inverse=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(N, N, kernel_size=5, padding=2, output_padding=1, stride=2),
                GDN(N, inverse=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(N, 32, kernel_size=5, padding=2, output_padding=1, stride=2),
            )
        )

    def forward(self, x, num_downsamples):
        # Use only the first 'num_downsamples' downsampling layers
        # print(f"ntc_forward: x={x.shape}, num_downsamples={num_downsamples}")
        for i in range(num_downsamples):
            x = self.encode[i](x)
            # print(f"{i} x={x.shape}")

        y_hat, y_likelihoods = self.entropy_bottleneck(x)

        for i in range(num_downsamples):
            y_hat = self.decode[i](y_hat)
            # print(f"{i} x={y_hat.shape}")

        # print(f"y_hat={y_hat.shape} type={type(y_hat)}")
        return y_hat, y_likelihoods

    #     self.entropy_bottleneck = EntropyBottleneck(N)
    #     self.encode = nn.Sequential(
    #         nn.Conv2d(32, N, stride=2, kernel_size=5, padding=2),
    #         GDN(N),
    #         nn.Conv2d(N, N, stride=2, kernel_size=5, padding=2),
    #         GDN(N),
    #         nn.Conv2d(N, N, stride=2, kernel_size=5, padding=2),
    #     )
    #
    #     self.decode = nn.Sequential(
    #         nn.ConvTranspose2d(N, N, kernel_size=5, padding=2, output_padding=1, stride=2),
    #         GDN(N, inverse=True),
    #         nn.ConvTranspose2d(N, N, kernel_size=5, padding=2, output_padding=1, stride=2),
    #         GDN(N, inverse=True),
    #         nn.ConvTranspose2d(N, 32, kernel_size=5, padding=2, output_padding=1, stride=2),
    #     )
    #
    # def forward(self, x):
    #    y = self.encode(x)
    #    y_hat, y_likelihoods = self.entropy_bottleneck(y)
    #    x_hat = self.decode(y_hat)
    #    return x_hat, y_likelihoods