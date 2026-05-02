import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from augmentations import SimCLRTransform
from dataset import ToadDataset, get_data_splits
from model import SimCLR
from loss import NTXentLoss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_history(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], "b-", label="Training Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], "r--", label="Validation Loss", linewidth=2)
    plt.title("SimCLR Training Trajectory", fontsize=16)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("NT-Xent Loss", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    set_seed(Config.SEED)

    # Create directories
    os.makedirs(Config.WEIGHTS_DIR, exist_ok=True)
    os.makedirs(Config.LOGS_DIR, exist_ok=True)

    # Get data splits (Split by Toad ID)
    train_files, val_files = get_data_splits(
        Config.DATA_DIR, val_split=Config.VAL_SPLIT, seed=Config.SEED
    )

    train_transform = SimCLRTransform(Config.IMG_SIZE)
    val_transform = SimCLRTransform(Config.IMG_SIZE)

    train_loader = DataLoader(
        ToadDataset(train_files, transform=train_transform),
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        ToadDataset(val_files, transform=val_transform),
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        num_workers=Config.NUM_WORKERS,
    )

    model = SimCLR(out_dim=Config.EMBEDDING_DIM).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Linear Warmup + Cosine Annealing
    def lr_lambda(epoch):
        if epoch < Config.WARMUP_EPOCHS:
            return float(epoch + 1) / float(max(1, Config.WARMUP_EPOCHS))
        return 0.5 * (
            1.0
            + np.cos(
                np.pi
                * (epoch - Config.WARMUP_EPOCHS)
                / (Config.EPOCHS - Config.WARMUP_EPOCHS)
            )
        )

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = NTXentLoss(Config.BATCH_SIZE, Config.TEMPERATURE, Config.DEVICE)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    print(f"Starting Training on {Config.DEVICE}...")
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS} [Train]")
        for x_i, x_j in pbar:
            x_i, x_j = x_i.to(Config.DEVICE), x_j.to(Config.DEVICE)
            optimizer.zero_grad()
            _, z_i = model(x_i)
            _, z_j = model(x_j)
            loss = criterion(z_i, z_j)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS} [Val]")
            for x_i, x_j in vbar:
                x_i, x_j = x_i.to(Config.DEVICE), x_j.to(Config.DEVICE)
                _, z_i = model(x_i)
                _, z_j = model(x_j)
                loss = criterion(z_i, z_j)
                val_loss += loss.item()
                vbar.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        curr_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch[{epoch + 1}/{Config.EPOCHS}] LR: {curr_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(Config.WEIGHTS_DIR, "best_simclr.pth"),
            )
            print("--> Best Model Saved")
        else:
            patience_counter += 1
            print(
                f"--> Early Stopping Counter: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    # Save final backbone
    torch.save(
        model.backbone.state_dict(),
        os.path.join(Config.WEIGHTS_DIR, "resnet50_backbone_final.pth"),
    )

    # Save history
    with open(os.path.join(Config.LOGS_DIR, "training_history.json"), "w") as f:
        json.dump(history, f)

    plot_history(history, os.path.join(Config.LOGS_DIR, "training_trajectory.png"))
    print("\nTraining Complete. Backbone saved.")


if __name__ == "__main__":
    main()
