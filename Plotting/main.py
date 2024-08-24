import DataLoader 
from Transform import transform 
from Plot import visualize_patient, visualize_slices, visualize_overlapped_slices, visualize_sagittal_slices
from Loss import DiceLoss
import torch

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

# Define root directory and other parameters
root_dir = 'C:/Users/Alejandro/Desktop/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
subject_index = 347
slice_number = 80  
slice_range = (80, 85)

# Create DataLoader instance
data_loader = DataLoader.NiftiDataset(root_dir, volumes=["t1c", "t1n", "t2f", "t2w"], seg_volume="seg", transform=transform())
# Get the subject at the specified index
subject = data_loader[subject_index]
# Execute visualization functions from Plot module
visualize_patient(subject, slice_number)
visualize_slices(subject, slice_range, seg_cmap='bone', other_cmap='bone')
visualize_overlapped_slices(subject, start_slice=10, end_slice=150, slice_spacing=5, cmap='gray', seg_cmap='rainbow')

slice_range2 = (30,200)  # Customize the range of slices to visualize
step = 15 # Customize the step size to skip slices
visualize_sagittal_slices(subject, 't1c', slice_range2, step, num_columns=4)


smooth_value = 1.0
weight_value = [0.3, 0.7]  # Example weights for binary classification


# Initialize an instance of the DiceLoss class
dice_loss = DiceLoss(smooth=smooth_value, weight=None).to(device)
