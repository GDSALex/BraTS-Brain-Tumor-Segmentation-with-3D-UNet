import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from Transform import transform
from DataLoader import NiftiDataset
from Unet3d2 import UNet3D
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import time
from Loss import CombinedLoss  

torch.cuda.empty_cache()
print(torch.cuda.get_device_name(0))

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
print(torch.version.cuda)

class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after 
    certain epochs (patience).
    """
    def __init__(self, patience=6, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        """
        Call method to check if validation loss has improved, else increments the counter.
        
        Args:
            val_loss (float): Current validation loss.
            model (nn.Module): Current model being trained.
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decrease.

        Args:
            val_loss (float): Current validation loss.
            model (nn.Module): Current model being trained.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def main():
    """
    Main function to train and validate the UNet3D model.
    """
    # Training parameters
    root_dir = '/workspace/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
    epochs = 100
    batch_size = 1  
    learning_rate = 0.001
    train_regions = ["t1c", "t1n", "t2f", "t2w"]
    seg_volume = "seg"

    # Create Dataset and split into training and validation subsets
    dataset = NiftiDataset(root_dir, volumes=train_regions, seg_volume=seg_volume, transform=transform())
    val_split = 0.1
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, Loss, and Optimizer
    model = UNet3D(in_channels=len(train_regions), out_channels=4).to(device)
    combined_loss = CombinedLoss.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    # EarlyStopping
    early_stopping = EarlyStopping(patience=6, verbose=True)

    # Training loop
    train_losses = []
    val_losses = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{epochs}", leave=False)

        for batch_idx, batch in progress_bar:
            if all(key in batch for key in train_regions) and seg_volume in batch:
                input_data = torch.cat([batch[key] for key in train_regions], dim=1).to(device)
                target_mask = batch[seg_volume].to(device)
                target_mask = target_mask.squeeze(1).long().to(device)
            else:
                print(f"Segmentation key '{seg_volume}' not found in batch or one of the train regions is missing. Batch keys: {batch.keys()}")
                continue

            optimizer.zero_grad()

            output_mask = model(input_data)
            loss = combined_loss(output_mask, target_mask)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * input_data.size(0)
            progress_bar.set_postfix({'Loss': f'{loss.item():.6f}'})
            progress_bar.update(1)

        epoch_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if all(key in batch for key in train_regions) and seg_volume in batch:
                    input_data = torch.cat([batch[key] for key in train_regions], dim=1).to(device, non_blocking=True)
                    target_mask = batch[seg_volume].to(device, non_blocking=True)
                    target_mask = target_mask.squeeze(1).long().to(device)
                else:
                    print(f"Segmentation key '{seg_volume}' not found in batch or one of the train regions is missing. Batch keys: {batch.keys()}")
                    continue

                output_mask = model(input_data)
                loss = combined_loss(output_mask, target_mask)
                val_loss += loss.item() * input_data.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch} learning rate: {current_lr}')

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        model_checkpoint_path = f"model_epoch_{epoch}.pth"
        torch.save(model.state_dict(), model_checkpoint_path)
        np.savetxt(f"train_losses_epoch_{epoch}.txt", np.array(train_losses))
        np.savetxt(f"val_losses_epoch_{epoch}.txt", np.array(val_losses))

        elapsed_time = time.time() - start_time
        remaining_time = (elapsed_time / epoch) * (epochs - epoch)
        print(f"Epoch [{epoch}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Validation Loss: {val_loss:.4f}, Elapsed Time: {elapsed_time:.2f}s, Remaining Time: {remaining_time:.2f}s")

    save_model_path = "/workspace/unet3d_model.pth"
    torch.save(model.state_dict(), save_model_path)

    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig("training_and_validation_losses.png")
    plt.show()

if __name__ == "__main__":
    main()
