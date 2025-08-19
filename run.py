"""run.py"""

import math
import argparse

import torch
from torch import nn
import torchio as tio
import matplotlib.pyplot as plt
from torch.utils.data import random_split

from model import SwinConvAE
from monai.losses import SSIMLoss
from dataloader import RandomPatchZeroOut, SSLPretextDataset

def parse_args():
    args = argparse.ArgumentParser(description="SwinConvAE Training")
    args.add_argument("--epochs", type=int, default=100)
    args.add_argument("--batch_size", type=int, default=4)
    args.add_argument("--dataset", type=str, default="dataset")
    return args.parse_args()

args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinConvAE(in_channels=1, out_channels=1, use_skip_connections=False).to(device)

amp = True 
lr = 3e-4
grad_clip = 1.0
num_workers = 4  
in_channels = 1
out_channels = 1
weight_decay = 1e-4
root_dir = args.dataset


base_transforms = tio.Compose([
    tio.RescaleIntensity(out_min_max=(0, 99.5), include=['mri']),
    tio.Resample((3, 3, 3), include=['mri']),
    tio.CropOrPad((64, 64, 64), include=['mri']),
])

mask_transform = RandomPatchZeroOut(
    patch_size=20,
    mask_ratio=0.75,
    include=['mri'],
    zero_value=0.0,
)

full_ds = SSLPretextDataset(
    root_dir=root_dir,
    base_transform=base_transforms,
    mask_transform=mask_transform,
)

val_len = max(1, int(0.1 * len(full_ds)))
train_len = len(full_ds) - val_len
train_ds, val_ds = random_split(full_ds, [train_len, val_len])

train_loader = tio.data.SubjectsLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
val_loader   = tio.data.SubjectsLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

criterion_mse = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

def psnr(mse_val: float, data_range: float = 1.0):
    if mse_val <= 1e-12:
        return 99.0
    return 10.0 * math.log10((data_range ** 2) / mse_val)

def run_one_epoch(model, train_loader, val_loader, optimizer, criterion_mse, epoch, best_val):
    model.train()
    running_loss = 0.0
    save_path = "SwinConvAE_SSL.pt"

    scaler = torch.cuda.amp.GradScaler(enabled=amp)


    for batch in train_loader:
        clean  = batch["clean"].float().to(device)   # target
        masked = batch["masked"].float().to(device)  # input

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            recon = model(masked)                 
            # Removed: loss_ssim = criterion_ssim(recon, clean)
            loss_mse = criterion_mse(recon, clean)
            loss = loss_mse

        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * masked.size(0)

    train_epoch_loss = running_loss / len(train_loader.dataset)

    # --- Validation --- #
    model.eval()
    val_running = 0.0
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
        for batch in val_loader:
            clean  = batch["clean"].float().to(device)
            masked = batch["masked"].float().to(device)
            recon = model(masked)
            val_loss = criterion_mse(recon, clean)
            val_running += val_loss.item() * masked.size(0)

    val_epoch_loss = val_running / len(val_loader.dataset)
    scheduler.step()

    train_psnr = psnr(train_epoch_loss, data_range=99.5) # Adjusted data_range
    val_psnr   = psnr(val_epoch_loss, data_range=99.5)   # Adjusted data_range

    print(f"[{epoch:03d}/{args.epochs}] "
          f"train MSE: {train_epoch_loss:.6f} (PSNR {train_psnr:.2f} dB) | "
          f"val MSE: {val_epoch_loss:.6f} (PSNR {val_psnr:.2f} dB) | "
          f"lr: {scheduler.get_last_lr()[0]:.2e}")

    if epoch % 5 == 0:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title('Clean')
        plt.imshow(clean[0, 0, ..., 32].detach().cpu(), cmap='gray')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.title('Masked')
        plt.imshow(masked[0, 0, ..., 32].detach().cpu(), cmap='gray')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.title('Reconstruction')
        plt.imshow(recon[0, 0, ..., 32].detach().cpu(), cmap='gray')
        plt.axis('off')
        
        plt.savefig(f"epoch_{epoch:03d}_recon.png")
        plt.close()

    if val_epoch_loss < best_val:
        best_val = val_epoch_loss
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_mse": best_val,
            "config": { 
                "in_channels": in_channels,
                "out_channels": out_channels,
                "use_skip_connections": model.use_skip_connections, 
                "mask_ratio": mask_transform.mask_ratio, 
                "patch_size": mask_transform.patch_size[0], 
            }
        }, save_path)
        print(f"✔ Saved new best to {save_path} (val MSE {best_val:.6f})")
    
    return best_val

if __name__ == "__main__":
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        best_val_loss = run_one_epoch(model, train_loader, val_loader, optimizer, criterion_mse, epoch, best_val_loss)
    print("Training finished.")
