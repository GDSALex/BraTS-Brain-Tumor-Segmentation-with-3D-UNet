import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from Transform import transform
from DataLoader import NiftiDataset
from Loss import DiceLoss
from Unet3d2 import UNet3D
import numpy as np
from tqdm import tqdm
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
from scipy.spatial.distance import directed_hausdorff
import napari 

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

# Define root directories
train_root_dir = 'C:/Users/Alejandro/Desktop/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
val_root_dir = 'C:/Users/Alejandro/Desktop/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData'

# Define modes
use_training_data = True  # Set to True to use training data
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
validation_split = 0.02
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
model.load_state_dict(torch.load('C:/Users/Alejandro/Desktop/BraTS Challenge CODE/4rd Training/model_epoch_40.pth'))
model.eval()

# Define loss functions
dice_loss = DiceLoss().to(device)

#cross_entropy_loss = nn.CrossEntropyLoss().to(device)

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
val_hausdorff_distances = []

# Lists to store subject images and masks for visualization
images = []
masks = []
predicted_masks = []

def calculate_dice_score(pred, target, smooth=1e-7):
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

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


def prepare_data_for_napari(patient_index):
    # Fetch the 4 MRI modalities for the selected patient
    t2f_volume = images[patient_index][0, 0, :, :, :]
    t2w_volume = images[patient_index][0, 1, :, :, :]
    t1c_volume = images[patient_index][0, 2, :, :, :]
    t1n_volume = images[patient_index][0, 3, :, :, :]
    
    # Fetch the predicted and ground truth masks
    predicted_mask = predicted_masks[patient_index][0]
    ground_truth_mask = masks[patient_index][0]
    
    return t2f_volume, t2w_volume, t1c_volume, t1n_volume, predicted_mask, ground_truth_mask

# Function to visualize patient data in napari
def visualize_with_napari(patient_index):
    # Prepare data
    t2f_volume, t2w_volume, t1c_volume, t1n_volume, predicted_mask, ground_truth_mask = prepare_data_for_napari(patient_index)

    # Start a napari viewer
    viewer = napari.Viewer()

    # Add MRI modalities to the viewer
    viewer.add_image(t2f_volume, name='T2f Modality', colormap='gray')
    viewer.add_image(t2w_volume, name='T2w Modality', colormap='gray')
    viewer.add_image(t1c_volume, name='T1c Modality', colormap='gray')
    viewer.add_image(t1n_volume, name='T1n Modality', colormap='gray')

    # Add predicted and ground truth masks to the viewer
    viewer.add_labels(predicted_mask, name='Predicted Mask', color={0: 'black', 1: 'blue', 2: 'green', 3: 'red'})
    viewer.add_labels(ground_truth_mask, name='Ground Truth Mask', color={0: 'black', 1: 'blue', 2: 'green', 3: 'red'})

    # Show the viewer
    napari.run()

# Randomly select a patient index
patient_index = np.random.randint(len(images))

# Visualize the selected patient's data with napari
visualize_with_napari(patient_index)
