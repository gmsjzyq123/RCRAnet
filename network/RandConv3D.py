import torch
import torch.nn.functional as F
import numpy as np

class RandConv3DBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, batch_size=64, device='cuda', sigma_d=0.2, input_dim=32):
        super(RandConv3DBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.device = device
        self.batch_size = batch_size
        self.sigma_d = sigma_d

        # For contrast diversification
        self.sigma_beta = 0.5
        self.sigma_alpha = 0.5

        # Sampled from N(n, sigma**2)
        self.gamma = torch.normal(mean=0, std=0.25, size=()).to(device)
        self.beta = torch.normal(mean=0, std=0.25, size=()).to(device)

        self.sigma_w = 1.0 / np.sqrt(kernel_size**3 * in_channels)
        self.weight = torch.nn.Parameter(torch.normal(mean=0, std=self.sigma_w, size=(out_channels, in_channels, kernel_size, kernel_size, kernel_size)).to(device))

        # Apply a Gaussian filter to the weights for smoothing
        b_g = 1.0
        epsilon = 1e-2
        sigma_g = np.random.uniform(epsilon, b_g)
        # gaussian_filter = torch.tensor([[[np.exp(-((i-1)**2 + (j-1)**2 + (k-1)**2) / (2 * sigma_g**2)) for i in range(kernel_size)] for j in range(kernel_size)] for k in range(kernel_size)], dtype=torch.float, device=device)
        # 计算归一化系数
        norm_factor = (2 * np.pi * sigma_g ** 2) ** (3 / 2)
        # 生成高斯滤波器
        gaussian_filter = torch.tensor([[[np.exp(
            -((i - 1) ** 2 + (j - 1) ** 2 + (k - 1) ** 2) / (2 * sigma_g ** 2)) / norm_factor for i in
                                          range(kernel_size)] for j in range(kernel_size)] for k in range(kernel_size)],
                                       dtype=torch.float, device=device)
        self.weight = torch.nn.Parameter(self.weight * gaussian_filter)

    def forward(self, x):
        # Perform a regular 3D convolution without deformable offsets
        conv_out = F.conv3d(x, self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation)

        # Apply the normalization and tanh activation
        standardized = (conv_out - conv_out.mean(dim=(2, 3, 4), keepdim=True)) / (conv_out.std(dim=(2, 3, 4), keepdim=True) + 1e-8)
        affined = standardized * self.gamma + self.beta
        out = F.tanh(affined)
        return out

class ProgRandConv3DBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, l_max=10, device='cuda', batch_size=64, sigma_d=0.2, input_dim=32):
        super(ProgRandConv3DBlock, self).__init__()
        self.n_layers = np.random.randint(1, l_max+1)
        # self.n_layers = l_max
        self.lyr = RandConv3DBlock(in_channels, out_channels, kernel_size, device=device, batch_size=batch_size, sigma_d=sigma_d, input_dim=input_dim).to(device)

    def forward(self, x):
        for _ in range(self.n_layers):
            x = self.lyr(x)
        return x
