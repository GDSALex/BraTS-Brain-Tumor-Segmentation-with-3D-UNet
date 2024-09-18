import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from Transform import transform
from DataLoader import NiftiDataset
from Loss import CombinedLoss
from Loss import DiceLoss
from Unet3d2 import UNet3D
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
from scipy.spatial.distance import directed_hausdorff

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

# Define root directories
train_root_dir = '/workspace/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
val_root_dir = '/workspace/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData'

# Define modes
use_training_data = False  # Set to True to use training data
apply_transforms = True  # Set to True to apply transforms

# Select appropriate dataset based on the mode
if use_training_data:
    if apply_transforms:
        dataset = NiftiDataset(train_root_dir, volumes=["t1c", "t1n", "t2f", "t2w"], seg_volume="seg", transform=transform())
    else:
        dataset = NiftiDataset(train_root_dir, volumes=["t1c", "t1n", "t2f", "t2w"], seg_volume="seg")
else:
    if apply_transforms:
        dataset = NiftiDataset(val_root_dir, volumes=["t1c", "t1n", "t2f", "t2w"], seg_volume="seg", transform=transform())
    else:
        dataset = NiftiDataset(val_root_dir, volumes=["t1c", "t1n", "t2f", "t2w"], seg_volume="seg")

# Define subset for validation
validation_split = 0.8   
shuffle_dataset = True
random_seed = 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

validation_indices = indices[:split]

# Creating PT data samplers and loaders:
validation_sampler = SubsetRandomSampler(validation_indices)
data_loader = DataLoader(dataset, batch_size=1, sampler=validation_sampler)

# Load the trained model
model = UNet3D(in_channels=4, out_channels=4).to(device)
model.load_state_dict(torch.load('C:/Users/Alejandro/Desktop/BraTS Challenge CODE/4rd training/model_epoch_40.pth'))
model.eval()

# Define loss functions
dice_loss = DiceLoss().to(device)


# Lists to store validation metrics
val_dice_scores = []
val_loss_values = []
ncr_dice_scores = []
ed_dice_scores = []
et_dice_scores = []
val_iou_scores = []
val_precision_scores = []
val_recall_scores = []
val_f1_scores = []
ncr_f1_scores = []
ed_f1_scores = []
et_f1_scores = []
val_hausdorff_distances = []
ncr_hd95_scores = []
ed_hd95_scores = []
et_hd95_scores = []

# Lists to store subject images and masks for visualization
images = []
masks = []
predicted_masks = []

def calculate_dice_score(pred, target, smooth=1e-7):
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def calculate_hd95(output_mask_points, target_mask_points):
    # Compute the directed Hausdorff distance from output to target and vice versa
    distances_pred_to_gt = [directed_hausdorff(output_mask_points, target_mask_points)[0]]
    distances_gt_to_pred = [directed_hausdorff(target_mask_points, output_mask_points)[0]]

    # Combine distances
    all_distances = distances_pred_to_gt + distances_gt_to_pred

    # Compute the 95th percentile Hausdorff Distance (HD95)
    hd95 = np.percentile(all_distances, 95)

    return hd95

# Validation loop
with torch.no_grad():
    for batch in tqdm(data_loader, desc="Validating"):
        input_data = torch.cat((batch['t1c'], batch['t1n'], batch['t2f'], batch['t2w']), dim=1).to(device, non_blocking=True)
        if use_training_data:
            target_mask = batch['seg'].to(device, dtype=torch.long, non_blocking=True).squeeze(1)
        else:
            target_mask = batch['seg'].to(device, dtype=torch.long, non_blocking=True).squeeze(1) if 'seg' in batch else None

        # Forward pass
        output_mask = model(input_data)

        if target_mask is not None:
            # Calculate loss
            dice_loss_value = dice_loss(output_mask, target_mask)
            #cross_entropy_loss_value = cross_entropy_loss(output_mask, target_mask)
            loss = dice_loss_value #+ cross_entropy_loss_value

            # Compute Dice score
            output_mask_class = (output_mask.argmax(dim=1)).float()

            ncr_dice = calculate_dice_score((output_mask_class == 1).float(), (target_mask == 1).float()).cpu().numpy()
            ed_dice = calculate_dice_score((output_mask_class == 2).float(), (target_mask == 2).float()).cpu().numpy()
            et_dice = calculate_dice_score((output_mask_class == 3).float(), (target_mask == 3).float()).cpu().numpy()


            ncr_dice_scores.append(ncr_dice)
            ed_dice_scores.append(ed_dice)
            et_dice_scores.append(et_dice)

            ncr_f1 = f1_score((target_mask == 1).cpu().numpy().flatten(), (output_mask_class == 1).cpu().numpy().flatten())
            ed_f1 = f1_score((target_mask == 2).cpu().numpy().flatten(), (output_mask_class == 2).cpu().numpy().flatten())
            et_f1 = f1_score((target_mask == 3).cpu().numpy().flatten(), (output_mask_class == 3).cpu().numpy().flatten())
            
            ncr_f1_scores.append(ncr_f1)
            ed_f1_scores.append(ed_f1)
            et_f1_scores.append(et_f1)

            # Ensure masks are flattened for metric calculations
            output_mask_np = output_mask_class.cpu().numpy().flatten()
            target_mask_np = target_mask.cpu().numpy().flatten()

            # Reshape for directed_hausdorff
            output_mask_points = np.column_stack(np.where(output_mask_class.cpu().numpy() > 0))
            target_mask_points = np.column_stack(np.where(target_mask.cpu().numpy() > 0))

            # Compute other metrics
            iou_score = jaccard_score(target_mask_np, output_mask_np, average='macro')
            precision = precision_score(target_mask_np, output_mask_np, average='macro', zero_division=0)
            recall = recall_score(target_mask_np, output_mask_np, average='macro')
            f1 = f1_score(target_mask_np, output_mask_np, average='macro')
            hausdorff_dist = max(directed_hausdorff(output_mask_points, target_mask_points)[0],
                                 directed_hausdorff(target_mask_points, output_mask_points)[0])
            
            for region, region_name in zip([1, 2, 3], ["NCR", "ED", "ET"]):
                # Extract boundary points for the specific region (NCR = 1, ED = 2, ET = 3)
                output_mask_points = np.column_stack(np.where((output_mask_class == region).cpu().numpy()))
                target_mask_points = np.column_stack(np.where((target_mask == region).cpu().numpy()))

                # Only calculate HD95 if there are points in both the prediction and ground truth masks for the region
                if len(output_mask_points) > 0 and len(target_mask_points) > 0:
                    hd95 = calculate_hd95(output_mask_points, target_mask_points)

                    # Store HD95 for the corresponding subregion
                    if region == 1:
                        ncr_hd95_scores.append(hd95)
                    elif region == 2:
                        ed_hd95_scores.append(hd95)
                    elif region == 3:
                        et_hd95_scores.append(hd95)

            # Store validation metrics
            val_loss_values.append(loss.item())
            val_iou_scores.append(iou_score)
            val_precision_scores.append(precision)
            val_recall_scores.append(recall)
            val_f1_scores.append(f1)
            val_hausdorff_distances.append(hausdorff_dist)

            # Store masks for visualization
            masks.append(target_mask.cpu().numpy())

        # Store images and predicted masks for visualization
        images.append(input_data.cpu().numpy())
        predicted_masks.append(output_mask.argmax(dim=1).cpu().numpy())

# Calculate average validation metrics if using training data
if use_training_data:
    avg_val_loss = np.mean(val_loss_values)
    avg_ncr_dice = np.mean(ncr_dice_scores)
    avg_ed_dice = np.mean(ed_dice_scores)
    avg_et_dice = np.mean(et_dice_scores)
    avg_iou_score = np.mean(val_iou_scores)
    avg_precision_score = np.mean(val_precision_scores)
    avg_recall_score = np.mean(val_recall_scores)
    avg_f1_score = np.mean(val_f1_scores)
    avg_hausdorff_distance = np.mean(val_hausdorff_distances)
    avg_ncr_f1 = np.mean(ncr_f1_scores)
    avg_et_f1 = np.mean(et_f1_scores)
    avg_ed_f1 = np.mean(ed_f1_scores)
    avg_ncr_hd95 = np.mean(ncr_hd95_scores) if ncr_hd95_scores else float('nan')
    avg_ed_hd95 = np.mean(ed_hd95_scores) if ed_hd95_scores else float('nan')
    avg_et_hd95 = np.mean(et_hd95_scores) if et_hd95_scores else float('nan')

    # Save validation metrics to a text file
    with open("validation_metrics.txt", "w") as file:
        file.write(f'Average Validation Loss: {avg_val_loss:.4f}\n')
        file.write(f'Average NCR Dice Score: {avg_ncr_dice:.4f}\n')
        file.write(f'Average ED Dice Score: {avg_ed_dice:.4f}\n')
        file.write(f'Average ET Dice Score: {avg_et_dice:.4f}\n')
        file.write(f'Average IoU Score: {avg_iou_score:.4f}\n')
        file.write(f'Average Precision: {avg_precision_score:.4f}\n')
        file.write(f'Average Recall: {avg_recall_score:.4f}\n')
        file.write(f'Average F1 Score: {avg_f1_score:.4f}\n')
        file.write(f'Average Hausdorff Distance: {avg_hausdorff_distance:.4f}\n')
        file.write(f'Average NCR F1: {avg_ncr_f1:.4f}\n')
        file.write(f'Average ET F1: {avg_et_f1:.4f}\n')
        file.write(f'Average ED F1: {avg_ed_f1:.4f}\n')
        file.write(f'Average NCR HD95: {avg_ncr_hd95:.4f}\n')
        file.write(f'Average ED HD95: {avg_ed_hd95:.4f}\n')
        file.write(f'Average ET HD95: {avg_et_hd95:.4f}\n')



slice_number = 85  # Adjust this number to select different slices

# Ensure slice_number is within bounds
for i in range(len(images)):
    print(f"Image {i} shape: {images[i].shape}")
    print(f"Predicted mask {i} shape: {predicted_masks[i].shape}")
    num_slices = images[i].shape[2]  
    if slice_number >= num_slices:
        slice_number = num_slices // 2  

# Plot some random images with their masks
num_samples = 4
sample_indices = np.random.choice(len(images), num_samples, replace=False)

plt.figure(figsize=(20, 5*num_samples))
for i, index in enumerate(sample_indices, 1):
    for channel in range(images[index].shape[1]):
        plt.subplot(num_samples, images[index].shape[1] + 3, (i-1) * (images[index].shape[1] + 3) + channel + 1)
        plt.imshow(images[index][0, channel, :, :, slice_number], cmap='gray')
        plt.axis('off')
        plt.title(f'Input Image (Channel {channel})')

    plt.subplot(num_samples, images[index].shape[1] + 3, (i-1) * (images[index].shape[1] + 3) + images[index].shape[1] + 1)
    plt.imshow(predicted_masks[index][0, :, :, slice_number], cmap='gray')
    plt.axis('off')
    plt.title('Predicted Mask')

    if use_training_data or target_mask is not None:
        plt.subplot(num_samples, images[index].shape[1] + 3, (i-1) * (images[index].shape[1] + 3) + images[index].shape[1] + 2)
        plt.imshow(masks[index][0, :, :, slice_number], cmap='gray')
        plt.axis('off')
        plt.title('Ground Truth Mask')

    # Add labels
    legend_labels = {
        1: 'NCR',
        2: 'ED',
        3: 'ET'
    }
    for value, label in legend_labels.items():
        plt.scatter([], [], color=plt.cm.gray(value / 3), label=label)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

plt.tight_layout()
plt.show()

# Plot validation losses if using training data
if use_training_data:
    plt.plot(range(1, len(val_loss_values) + 1), val_loss_values, label='Validation Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.show()
