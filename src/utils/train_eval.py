"""
Training and evaluation utility functions for the Wavelet Packet Autoencoder.
"""
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.models.wavelet_packet_ae import wavelet_packet_sparsity_loss

def schedule_lambda_gamma(epoch, total_epochs, lambda_start, lambda_end, gamma_start, gamma_end):
    frac = (epoch - 1) / max(1, total_epochs - 1)
    lambda_e = lambda_start + (lambda_end - lambda_start) * frac
    gamma_e = gamma_start + (gamma_end - gamma_start) * frac
    lambda_e = max(0.0, min(1.0, lambda_e))
    gamma_e = max(0.0, min(1.0, gamma_e))
    sum_lg = lambda_e + gamma_e
    if sum_lg < 1.0:
        scale = 1.0 / (sum_lg + 1e-12)
        lambda_e *= scale
        gamma_e *= scale
        lambda_e = min(lambda_e, 1.0)
        gamma_e = min(gamma_e, 1.0)
    return lambda_e, gamma_e

def log_audio_and_metrics(model, loader, writer, epoch, n_samples=2, sr=16000):
    model.eval()
    metrics_pesq = []
    metrics_stoi = []
    for clean_batch, noisy_batch, _ in loader:
        clean_batch = clean_batch.to("cpu")
        noisy_batch = noisy_batch.to("cpu")
        output, _ = model(noisy_batch)
        B = clean_batch.shape[0]
        chosen = min(n_samples, B)
        for i in range(chosen):
            clean_np = clean_batch[i].cpu().numpy()
            out_np = output[i].cpu().detach().numpy()
            ref_1d = clean_np[0].astype("float32")
            deg_1d = out_np[0].astype("float32")
            length = min(len(ref_1d), len(deg_1d))
            ref_1d = ref_1d[:length]
            deg_1d = deg_1d[:length]
            try:
                from pesq import pesq
                pesq_val = pesq(sr, ref_1d, deg_1d, 'wb')
            except Exception:
                pesq_val = 0.0
            from pystoi import stoi
            stoi_val = stoi(ref_1d, deg_1d, sr, extended=False)
            metrics_pesq.append(pesq_val)
            metrics_stoi.append(stoi_val)
            writer.add_scalar(f"SpeechMetrics/PESQ_sample_{i}", pesq_val, epoch)
            writer.add_scalar(f"SpeechMetrics/STOI_sample_{i}", stoi_val, epoch)
            writer.add_audio(f"Audio_Denoised/sample_{i}", torch.from_numpy(deg_1d).unsqueeze(0), epoch, sample_rate=sr)
            writer.add_audio(f"Audio_Clean/sample_{i}", torch.from_numpy(ref_1d).unsqueeze(0), epoch, sample_rate=sr)
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(ref_1d, label="Clean")
            ax.plot(deg_1d, label="Denoised", alpha=0.7)
            ax.legend()
            ax.set_title(f"Waveforms sample {i}, epoch={epoch}")
            writer.add_figure(f"WaveformPlot/sample_{i}", fig, global_step=epoch)
            plt.close(fig)
        break
    if metrics_pesq:
        writer.add_scalar("SpeechMetrics/PESQ_avg", sum(metrics_pesq)/len(metrics_pesq), epoch)
        writer.add_scalar("SpeechMetrics/STOI_avg", sum(metrics_stoi)/len(metrics_stoi), epoch)
    writer.flush()

def log_LAHT_and_wavelets(writer, model, epoch):
    # Log histograms for wavelet kernels.
    for lev, mod in enumerate(model.kernelsG_):
        writer.add_histogram(f"KernelsG/Level_{lev+1}", mod.kernel.cpu().detach(), epoch)
    if hasattr(model, 'kernelsH_') and (model.kernelsH_ is not model.kernelsG_):
        for lev, mod in enumerate(model.kernelsH_):
            writer.add_histogram(f"KernelsH/Level_{lev+1}", mod.kernel.cpu().detach(), epoch)
    # Log LAHT parameter scalars.
    for lev in range(model.level):
        laht_mod = model.HardThresholdAssymH[lev]
        writer.add_scalar(f"LAHT_Params/Level_{lev+1}_thrP", laht_mod.thrP.item(), epoch)
        writer.add_scalar(f"LAHT_Params/Level_{lev+1}_thrN", laht_mod.thrN.item(), epoch)
        writer.add_scalar(f"LAHT_Params/Level_{lev+1}_alpha", laht_mod.alpha.item(), epoch)
        writer.add_scalar(f"LAHT_Params/Level_{lev+1}_beta", laht_mod.beta.item(), epoch)
    # Log LAHT function curves.
    fig, ax = plt.subplots(figsize=(6, 4))
    x_vals = torch.linspace(-2, 2, 200).to("cpu")
    for lev in range(model.level):
        y_vals = model.HardThresholdAssymH[lev](x_vals).detach().cpu().numpy()
        ax.plot(x_vals.cpu().numpy(), y_vals, label=f"Level {lev+1}")
    ax.set_title(f"LAHT Curves Epoch {epoch}")
    ax.set_xlabel("x")
    ax.set_ylabel("LAHT(x)")
    ax.legend()
    ax.grid(True)
    writer.add_figure("LAHT_Curves", fig, global_step=epoch)
    plt.close(fig)
    writer.flush()

def train(model, loader, optimizer, epoch, lambda_, gamma_, log_interval=10, writer=None):
    model.train()
    train_loss = 0
    for batch_idx, (clean_batch, noisy_batch, _) in enumerate(loader):
        clean_batch = clean_batch.to("cpu")
        noisy_batch = noisy_batch.to("cpu")
        optimizer.zero_grad()
        output, detail_list = model(noisy_batch)
        loss = wavelet_packet_sparsity_loss(output, clean_batch, detail_list, lambda_, gamma_)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(clean_batch)}/{len(loader.dataset)} "
                  f"({100.*batch_idx/len(loader):.0f}%)] Loss: {loss.item():.6f}")
    train_loss /= len(loader)
    if writer:
        log_LAHT_and_wavelets(writer, model, epoch)
    writer.add_scalars("Loss/Epoch", {"Train": train_loss}, epoch)
    writer.add_scalars("Schedules/Epoch", {"Gamma": gamma_, "Lambda": lambda_}, epoch)
    writer.flush()
    return train_loss

def validate(model, loader, lambda_, gamma_, epoch=None, writer=None):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for clean_batch, noisy_batch, _ in loader:
            clean_batch = clean_batch.to("cpu")
            noisy_batch = noisy_batch.to("cpu")
            output, detail_list = model(noisy_batch)
            loss = wavelet_packet_sparsity_loss(output, clean_batch, detail_list, lambda_, gamma_)
            val_loss += loss.item()
    val_loss /= len(loader)
    print(f"\nValidation set: Average loss: {val_loss:.4f}\n")
    if writer is not None and epoch is not None:
        writer.add_scalars("Loss/Epoch", {"Val": val_loss}, epoch)
        writer.add_scalars("Schedules/Epoch", {"Gamma": gamma_, "Lambda": lambda_}, epoch)
    writer.flush()
    return val_loss

def plot_threshold_functions(model, level, title, x_range=(-2,2)):
    x = torch.linspace(x_range[0], x_range[1], 1000).to("cpu")
    plt.figure(figsize=(10, 4))
    for lev in range(level):
        y_cur = model.HardThresholdAssymH[lev](x).detach().cpu().numpy()
        plt.plot(x.cpu().numpy(), y_cur, label=f"Level {lev+1}")
    plt.title(title)
    plt.xlabel("Input")
    plt.ylabel("LAHT(x)")
    plt.legend()
    plt.grid(True)
    plt.show()
