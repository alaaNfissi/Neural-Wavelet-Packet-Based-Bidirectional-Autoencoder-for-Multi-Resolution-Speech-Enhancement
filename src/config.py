# Configuration file for data paths and hyperparameters.
import os

# Data directories (override with environment variables or update as needed)
CLEAN_TRAIN_DIR = os.getenv("CLEAN_TRAIN_DIR", "/mnt/bigdisk/data/alaa/Wavelets/Valentini_VoiceBank_DEMAND/clean_trainset_28spk_wav")
NOISY_TRAIN_DIR = os.getenv("NOISY_TRAIN_DIR", "/mnt/bigdisk/data/alaa/Wavelets/Valentini_VoiceBank_DEMAND/noisy_trainset_28spk_wav")
CLEAN_TEST_DIR  = os.getenv("CLEAN_TEST_DIR",  "/mnt/bigdisk/data/alaa/Wavelets/Valentini_VoiceBank_DEMAND/clean_testset_wav")
NOISY_TEST_DIR  = os.getenv("NOISY_TEST_DIR",  "/mnt/bigdisk/data/alaa/Wavelets/Valentini_VoiceBank_DEMAND/noisy_testset_wav")

# Sample rates
INPUT_SR = 48000
TARGET_SR = 16000

# Training hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
LOG_INTERVAL = 10

# Schedule parameters for loss
LAMBDA_START = 1.0
LAMBDA_END = 0.0
GAMMA_START = 0.0
GAMMA_END = 1.0

# Wavelet parameters
WAVELET = "db20"  # Using Daubechies 20
KERNEL_INIT = 20  # If integer, this will be used directly (else use string to load wavelet)
KERNELS_CONSTRAINT = "Bidirectional"  # Options: "Bidirectional", "Unidirectional"
TRAIN_HT = True
INIT_THP = 1.0
INIT_THN = 1.0
ALPHA = 10
BETA = 10

# Run name for TensorBoard logs (append timestamp)
RUN_NAME = f"WaveletExp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_Valentini"
