#!/usr/bin/env python3
"""
Main entry point for training and evaluating the Wavelet Packet Autoencoder.
"""
import datetime
import os

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from config import (CLEAN_TRAIN_DIR, NOISY_TRAIN_DIR, CLEAN_TEST_DIR, NOISY_TEST_DIR,
                    INPUT_SR, TARGET_SR, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
                    LOG_INTERVAL, LAMBDA_START, LAMBDA_END, GAMMA_START, GAMMA_END,
                    WAVELET, KERNEL_INIT, KERNELS_CONSTRAINT, TRAIN_HT, INIT_THP, INIT_THN, ALPHA, BETA, RUN_NAME)
from data.dataset import create_valentini_dataloaders
from models.wavelet_packet_ae import WaveletPacketAutoencoder
from utils.train_eval import schedule_lambda_gamma, train, validate, log_audio_and_metrics, log_LAHT_and_wavelets

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    # Create dataloaders
    train_loader, test_loader = create_valentini_dataloaders(
        CLEAN_TRAIN_DIR, NOISY_TRAIN_DIR, CLEAN_TEST_DIR, NOISY_TEST_DIR,
        batch_size=BATCH_SIZE, input_sr=INPUT_SR, target_sr=TARGET_SR
    )

    # Initialize wavelet kernel based on chosen wavelet
    import pywt
    wv = pywt.Wavelet(WAVELET)
    kernel_init = np.array(wv.filter_bank[0]) if isinstance(KERNEL_INIT, str) or KERNEL_INIT == "db20" else KERNEL_INIT

    model = WaveletPacketAutoencoder(
        stride=2,
        kernelInit=kernel_init,
        kernTrainable=True,
        level=15,
        kernelsConstraint=KERNELS_CONSTRAINT,
        initThP=INIT_THP,
        initThN=INIT_THN,
        trainHT=TRAIN_HT,
        alpha=ALPHA,
        beta=BETA
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter(log_dir=f"runs/{RUN_NAME}")

    train_losses, val_losses = [], []
    for epoch in range(1, NUM_EPOCHS + 1):
        lam_e, gam_e = schedule_lambda_gamma(epoch, NUM_EPOCHS, LAMBDA_START, LAMBDA_END, GAMMA_START, GAMMA_END)
        t_loss = train(model, train_loader, optimizer, epoch, lam_e, gam_e, log_interval=LOG_INTERVAL, writer=writer)
        v_loss = validate(model, test_loader, lam_e, gam_e, epoch=epoch, writer=writer)
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        # Log unified scalar charts
        writer.add_scalars("Loss/Epoch", {"Train": t_loss, "Val": v_loss}, epoch)
        writer.add_scalars("Schedules/Epoch", {"Gamma": gam_e, "Lambda": lam_e}, epoch)
        # Optionally log audio and metrics every 10 epochs
        if epoch % 10 == 0:
            log_audio_and_metrics(model, test_loader, writer, epoch, n_samples=3, sr=TARGET_SR)
    writer.close()

    # Plot training curves
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Train')
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Wavelet Packet Autoencoder Training (Valentini Dataset)")
    plt.grid(True)
    plt.show()

    # Plot threshold functions
    from utils.train_eval import plot_threshold_functions
    plot_threshold_functions(model, 15, "Threshold Functions", x_range=(-3, 3))

if __name__ == "__main__":
    main()
