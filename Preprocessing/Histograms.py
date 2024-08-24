import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torchio as tio
from scipy import stats
from concurrent.futures import ThreadPoolExecutor

# Paths to the NIfTI files
root_dir = '/workspace/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
seg_volume = "seg"
volume_types = ["t1c", "t1n", "t2f", "t2w", seg_volume]

# Define colors for each volume type
volume_colors = {
    "t1c": "red",
    "t1n": "green",
    "t2f": "blue",
    "t2w": "purple",
    seg_volume: "orange"
}

# Ensure paths are correct
def get_volume_paths(root_dir, volume):
    paths = []
    for patient_folder in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_folder)
        if os.path.isdir(patient_path):
            volume_path = os.path.join(patient_path, f'{patient_folder}-{volume}.nii.gz')
            if os.path.exists(volume_path):
                paths.append(volume_path)
    
    return paths

paths = {volume: get_volume_paths(root_dir, volume) for volume in volume_types}


# Function to plot histograms
def plot_histogram(axis, tensor, color, num_positions=100, label=None, alpha=0.05):
    values = tensor.numpy().ravel()
    kernel = stats.gaussian_kde(values)
    positions = np.linspace(values.min(), values.max(), num=num_positions)
    histogram = kernel(positions)
    kwargs = dict(linewidth=1, color=color, alpha=alpha)
    if label is not None:
        kwargs['label'] = label
    axis.plot(positions, histogram, **kwargs)

# Plot original histograms
def plot_original_histograms(paths):
    fig, ax = plt.subplots(dpi=100)
    def plot_volume_histogram(volume, path_list):
        color = volume_colors[volume]
        for path in tqdm(path_list, desc=f'Plotting original histograms for {volume}'):
            tensor = tio.ScalarImage(path).data
            plot_histogram(ax, tensor, color)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(plot_volume_histogram, volume, path_list) for volume, path_list in paths.items()]
        for future in futures:
            future.result()
    ax.set_xlim(-100, 2000)
    ax.set_ylim(0, 0.004)
    ax.set_title('Original histograms of all samples')
    ax.set_xlabel('Intensity')
    ax.grid()
    plt.show()

# Function to train histogram standardization for each volume type
def train_histogram_standardization(volume, paths, output_path):
    if len(paths) == 0:
        raise ValueError(f"No files found for volume {volume}")
    landmarks = tio.HistogramStandardization.train(paths, output_path=output_path)
    torch.save(landmarks, output_path)
    return landmarks

# Train histogram standardization and save landmarks
landmarks_paths = {volume: f'{volume}_landmarks.pt' for volume in volume_types if volume != seg_volume}
landmarks_dict = {}

with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    futures = []
    for volume, path_list in paths.items():
        if volume == seg_volume:
            continue
        output_path = landmarks_paths[volume]
        futures.append(executor.submit(train_histogram_standardization, volume, path_list, output_path))

    for future, volume in zip(futures, volume_types):
        if volume != seg_volume:
            landmarks_dict[volume] = future.result()

# Function to apply histogram standardization and plot histograms
def plot_standardized_histograms(paths, landmarks_dict):
    fig, ax = plt.subplots(dpi=100)
    def plot_volume_histogram(volume, path_list):
        histogram_transform = tio.ZNormalization(include=[volume])
        color = volume_colors[volume]
        for path in tqdm(path_list, desc=f'Plotting standardized histograms for {volume}'):
            image = tio.ScalarImage(path)
            subject = tio.Subject({volume: image})
            standardized = histogram_transform(subject)
            tensor = standardized[volume].data
            plot_histogram(ax, tensor, color)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(plot_volume_histogram, volume, path_list) for volume, path_list in paths.items() if volume != seg_volume]
        for future in futures:
            future.result()
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 0.02)
    ax.set_title('Intensity values of all samples after histogram standardization')
    ax.set_xlabel('Intensity')
    ax.grid()
    plt.show()

# Plot original histograms
plot_original_histograms(paths)

# Plot standardized histograms
plot_standardized_histograms(paths, landmarks_dict)
