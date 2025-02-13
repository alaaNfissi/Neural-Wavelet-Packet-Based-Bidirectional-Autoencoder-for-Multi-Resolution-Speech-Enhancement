"""
This module provides the transform pipeline.
"""
import torch
import torchaudio
from torch import nn
from src.config import INPUT_SR, TARGET_SR  # If needed

class MyTransformPipeline(nn.Module):
    """
    Resamples from input_freq to resample_freq and globally normalizes
    waveforms to [-1,1] using precomputed global_min and global_max.
    """
    def __init__(self, input_freq=48000, resample_freq=16000, global_min=None, global_max=None):
        super().__init__()
        self.input_freq = input_freq
        self.resample_freq = resample_freq
        self.global_min = global_min
        self.global_max = global_max
        if self.input_freq != self.resample_freq:
            self.resample = torchaudio.transforms.Resample(orig_freq=self.input_freq,
                                                            new_freq=self.resample_freq).to("cpu")
        else:
            self.resample = None

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.to("cpu")
        if self.resample is not None:
            waveform = self.resample(waveform)
        denom = (self.global_max - self.global_min)
        if denom > 0:
            waveform = 2.0 * (waveform - self.global_min) / denom - 1.0
        else:
            waveform = waveform - self.global_min
        return waveform.to("cpu")
