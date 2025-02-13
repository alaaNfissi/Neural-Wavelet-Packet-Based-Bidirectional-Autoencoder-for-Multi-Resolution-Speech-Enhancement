"""
Dataset module for loading and processing the Valentini VoiceBank-DEMAND data.
"""

import os
import torchaudio
import torch
from .utils import pad_sequence  # Alternatively, if you want to put pad_sequence in this file

def list_wav_files(folder):
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".wav")])

def matching_pairs(clean_dir, noisy_dir):
    clean_files = list_wav_files(clean_dir)
    noisy_files = list_wav_files(noisy_dir)
    if len(clean_files) != len(noisy_files):
        raise ValueError("Mismatch in number of clean vs. noisy files.")
    return clean_files, noisy_files

def compute_precise_mean_std(file_paths):
    sum_waveform = 0.0
    sum_squares  = 0.0
    total_samples = 0
    for path in file_paths:
        waveform, _ = torchaudio.load(path)
        sum_waveform += waveform.sum()
        sum_squares  += (waveform ** 2).sum()
        total_samples += waveform.numel()
    mean = sum_waveform / total_samples
    std = (sum_squares / total_samples - mean ** 2) ** 0.5
    return mean.item(), std.item()

def compute_global_min_max(file_paths, input_sr=48000, target_sr=16000):
    global_min = float("inf")
    global_max = float("-inf")
    resampler = None
    if input_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=input_sr, new_freq=target_sr)
    for path in file_paths:
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform[0].unsqueeze(0)
        if resampler is not None:
            waveform = resampler(waveform)
        wave_min = waveform.min().item()
        wave_max = waveform.max().item()
        if wave_min < global_min:
            global_min = wave_min
        if wave_max > global_max:
            global_max = wave_max
    return global_min, global_max

class ValentiniDataset(torch.utils.data.Dataset):
    def __init__(self, clean_files, noisy_files, transform=None):
        self.clean_files = clean_files
        self.noisy_files = noisy_files
        self.transform = transform

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_path = self.clean_files[idx]
        noisy_path = self.noisy_files[idx]
        clean_wave, sr_clean = torchaudio.load(clean_path)
        noisy_wave, sr_noisy = torchaudio.load(noisy_path)
        if self.transform:
            clean_wave = self.transform(clean_wave)
            noisy_wave = self.transform(noisy_wave)
        if clean_wave.shape[0] > 1:
            clean_wave = clean_wave[0].unsqueeze(0)
        if noisy_wave.shape[0] > 1:
            noisy_wave = noisy_wave[0].unsqueeze(0)
        return clean_wave, noisy_wave, sr_clean, "Valentini"

def pad_sequence(batch):
    clean_list = []
    noisy_list = []
    label_list = []
    for (clean_w, noisy_w, sr, label) in batch:
        clean_list.append(clean_w)
        noisy_list.append(noisy_w)
        label_list.append(label)
    # Pad along the time dimension
    clean_list = [w.t() for w in clean_list]
    noisy_list = [w.t() for w in noisy_list]
    clean_padded = torch.nn.utils.rnn.pad_sequence(clean_list, batch_first=True, padding_value=0.0).permute(0, 2, 1)
    noisy_padded = torch.nn.utils.rnn.pad_sequence(noisy_list, batch_first=True, padding_value=0.0).permute(0, 2, 1)
    return clean_padded, noisy_padded, label_list

def create_valentini_dataloaders(clean_train_dir, noisy_train_dir, clean_test_dir, noisy_test_dir, batch_size=16, input_sr=48000, target_sr=16000):
    from .dataset import matching_pairs, compute_global_min_max, ValentiniDataset
    train_clean, train_noisy = matching_pairs(clean_train_dir, noisy_train_dir)
    test_clean, test_noisy = matching_pairs(clean_test_dir, noisy_test_dir)
    all_train = train_clean + train_noisy
    global_min, global_max = compute_global_min_max(all_train, input_sr, target_sr)
    print(f"Global Min, Max => {global_min}, {global_max}")
    from src.utils.transform_pipeline import MyTransformPipeline
    transform = MyTransformPipeline(input_freq=input_sr, resample_freq=target_sr, global_min=global_min, global_max=global_max)
    train_dataset = ValentiniDataset(train_clean, train_noisy, transform=transform)
    test_dataset  = ValentiniDataset(test_clean, test_noisy, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=pad_sequence)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=pad_sequence)
    return train_loader, test_loader
