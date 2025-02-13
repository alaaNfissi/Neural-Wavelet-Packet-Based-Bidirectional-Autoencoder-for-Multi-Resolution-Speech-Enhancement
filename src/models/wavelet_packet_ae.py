"""
Wavelet Packet Autoencoder (WPAE) module.
Implements a recursive discrete wavelet packet transform (DWPT) autoencoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Kernel(nn.Module):
    def __init__(self, kernelInit, trainKern=True):
        super().__init__()
        self.trainKern = trainKern
        if isinstance(kernelInit, int):
            self.kernelSize = kernelInit
            self.kernel = nn.Parameter(torch.empty(self.kernelSize), requires_grad=trainKern)
            nn.init.normal_(self.kernel)
        elif isinstance(kernelInit, (list, np.ndarray, torch.Tensor)):
            self.kernelSize = len(kernelInit)
            if isinstance(kernelInit, np.ndarray):
                kernelInit = torch.tensor(kernelInit, dtype=torch.float32)
            self.kernel = nn.Parameter(kernelInit.view(self.kernelSize), requires_grad=trainKern)
        else:
            raise TypeError("kernelInit must be int, list, np.ndarray, or torch.Tensor")
    def forward(self, inputs=None):
        return self.kernel

class LowPassWave(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride
    def forward(self, inputs):
        input_signal, kernel = inputs
        kernel_size = kernel.size(-1)
        padding = (kernel_size - 1) // 2
        return F.conv1d(input_signal, kernel.view(1, 1, -1).to(input_signal.device),
                        padding=padding, stride=self.stride)

class HighPassWave(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride
    def initialize_qmfFlip(self, kernel_size, device):
        qmfFlip = torch.tensor([(-1)**i for i in range(kernel_size)], dtype=torch.float32)
        self.qmfFlip = nn.Parameter(qmfFlip.view(1, 1, -1), requires_grad=False).to(device)
    def forward(self, inputs):
        input_signal, kernel = inputs
        kernel_size = kernel.size(-1)
        if not hasattr(self, 'qmfFlip'):
            self.initialize_qmfFlip(kernel_size, input_signal.device)
        padding = (kernel_size - 1) // 2
        reversed_kernel = torch.flip(kernel.view(1, 1, -1), [2]).to(input_signal.device)
        adjusted_kernel = reversed_kernel * self.qmfFlip
        return F.conv1d(input_signal, adjusted_kernel, padding=padding, stride=self.stride)

class LowPassTrans(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride
    def forward(self, inputs):
        input_signal, kernel = inputs
        kernel_size = kernel.size(-1)
        padding = (kernel_size - 1) // 2
        output_padding = (input_signal.size(2) * self.stride - kernel.size(-1) + 1) % self.stride
        return F.conv_transpose1d(input_signal, kernel.view(1, 1, -1).to(input_signal.device),
                                  stride=self.stride, padding=padding, output_padding=output_padding)

class HighPassTrans(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride
    def initialize_qmfFlip(self, kernel_size, device):
        qmfFlip = torch.tensor([(-1)**i for i in range(kernel_size)], dtype=torch.float32)
        self.qmfFlip = nn.Parameter(qmfFlip.view(1, 1, -1), requires_grad=False).to(device)
    def forward(self, inputs):
        input_signal, kernel = inputs
        kernel_size = kernel.size(-1)
        if not hasattr(self, 'qmfFlip'):
            self.initialize_qmfFlip(kernel_size, input_signal.device)
        padding = (kernel_size - 1) // 2
        output_padding = (input_signal.size(2) * self.stride - kernel.size(-1) + 1) % self.stride
        reversed_kernel = torch.flip(kernel.view(1, 1, -1), [2]).to(input_signal.device)
        adjusted_kernel = reversed_kernel * self.qmfFlip
        return F.conv_transpose1d(input_signal, adjusted_kernel,
                                  stride=self.stride, padding=padding, output_padding=output_padding)

class HardThresholdAssym(nn.Module):
    def __init__(self, initThP=1.0, initThN=1.0, alpha=10, beta=10, trainBias=True):
        super().__init__()
        self.trainBias = trainBias
        self.initP = torch.tensor([initThP], dtype=torch.float32)
        self.initN = torch.tensor([initThN], dtype=torch.float32)
        self.alpha = torch.tensor([alpha], dtype=torch.float32)
        self.beta  = torch.tensor([beta], dtype=torch.float32)
        if torch.cuda.is_available():
            self.initP = self.initP.to(device)
            self.initN = self.initN.to(device)
            self.alpha = self.alpha.to(device)
            self.beta  = self.beta.to(device)
        self.thrP = nn.Parameter(self.initP, requires_grad=trainBias)
        self.thrN = nn.Parameter(self.initN, requires_grad=trainBias)
        self.alpha = nn.Parameter(self.alpha, requires_grad=trainBias)
        self.beta  = nn.Parameter(self.beta, requires_grad=trainBias)
    def forward(self, inputs):
        return inputs * (torch.sigmoid(self.alpha*(inputs - self.thrP)) +
                         torch.sigmoid(-self.beta*(inputs + self.thrN)))

class WaveletPacketAutoencoder(nn.Module):
    def __init__(self, stride=2, kernelInit=20, kernTrainable=True,
                 level=1, kernelsConstraint='Bidirectional',
                 initThP=1.0, initThN=1.0, trainHT=True, alpha=10, beta=10):
        super().__init__()
        self.level = level
        self.stride = stride
        self.alpha = alpha
        self.beta  = beta
        # Initialize kernels according to constraint.
        if kernelsConstraint == 'Bidirectional':
            self.kernelsG_ = nn.ModuleList([Kernel(kernelInit, trainKern=kernTrainable) for _ in range(level)])
            self.kernelsH_ = nn.ModuleList([Kernel(kernelInit, trainKern=kernTrainable) for _ in range(level)])
            self.kernelsGT_ = self.kernelsG_
            self.kernelsHT_ = self.kernelsH_
        elif kernelsConstraint == 'Unidirectional':
            self.kernelsG_ = nn.ModuleList([Kernel(kernelInit, trainKern=kernTrainable) for _ in range(level)])
            self.kernelsH_ = nn.ModuleList([Kernel(kernelInit, trainKern=kernTrainable) for _ in range(level)])
            self.kernelsGT_ = nn.ModuleList([Kernel(kernelInit, trainKern=kernTrainable) for _ in range(level)])
            self.kernelsHT_ = nn.ModuleList([Kernel(kernelInit, trainKern=kernTrainable) for _ in range(level)])
        else:
            raise ValueError("Invalid kernelsConstraint option")
        self.LowPassWave  = nn.ModuleList([LowPassWave(stride=self.stride) for _ in range(level)])
        self.HighPassWave = nn.ModuleList([HighPassWave(stride=self.stride) for _ in range(level)])
        self.LowPassTrans = nn.ModuleList([LowPassTrans(stride=self.stride) for _ in range(level)])
        self.HighPassTrans= nn.ModuleList([HighPassTrans(stride=self.stride) for _ in range(level)])
        self.HardThresholdAssymH = nn.ModuleList([
            HardThresholdAssym(initThP=initThP, initThN=initThN, alpha=self.alpha, beta=self.beta, trainBias=trainHT)
            for _ in range(level)
        ])
    def encode(self, x, current_level=0, detail_list=None):
        if detail_list is None:
            detail_list = []
        if current_level < self.level:
            low = self.LowPassWave[current_level]([x, self.kernelsG_[current_level]()])
            high = self.HighPassWave[current_level]([x, self.kernelsH_[current_level]()])
            high = self.HardThresholdAssymH[current_level](high)
            detail_list.append(high)
            low_enc, detail_list = self.encode(low, current_level + 1, detail_list)
            return low_enc, detail_list
        else:
            return x, detail_list
    def decode(self, coeff, current_level=0):
        if current_level < self.level:
            low_up = self.LowPassTrans[current_level]([coeff, self.kernelsGT_[current_level]()])
            high_up = self.HighPassTrans[current_level]([coeff, self.kernelsHT_[current_level]()])
            if low_up.size(2) != high_up.size(2):
                ms = min(low_up.size(2), high_up.size(2))
                low_up = low_up[:, :, :ms]
                high_up = high_up[:, :, :ms]
            return low_up + high_up
        else:
            return coeff
    def forward(self, x):
        encoded, detail_list = self.encode(x, current_level=0, detail_list=[])
        reconstructed = self.decode(encoded, current_level=0)
        if reconstructed.size(2) > x.size(2):
            reconstructed = reconstructed[:, :, :x.size(2)]
        elif reconstructed.size(2) < x.size(2):
            pad_len = x.size(2) - reconstructed.size(2)
            reconstructed = F.pad(reconstructed, (0, pad_len))
        return reconstructed, detail_list
