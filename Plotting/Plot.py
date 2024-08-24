

import matplotlib.pyplot as plt
import numpy as np 
from scipy.ndimage import rotate
import matplotlib.patches as patches


def visualize_patient(subject, slice_number):
    """
    Visualizes a specified slice of all volumes for a patient.

    Args:
        subject (tio.Subject): A dictionary containing loaded volumes and
            segmentation as torchio.ScalarImage or torchio.LabelMap objects.
        slice_number (int): The slice number to visualize.
    """

    volume_names = list(subject.keys())
    volumes = [subject[name].data.numpy() for name in volume_names]

    plt.figure(figsize=(15, 15))

    for i, volume_name in enumerate(volume_names, 1):
        plt.subplot(1, len(volume_names), i)
        volume = subject[volume_name]
        volume_array = volume.data.numpy()  
        volume_array = np.squeeze(volume_array)  
        volume_slice = volume_array[..., slice_number]
        if volume_name == 'seg':
            plt.imshow(volume_slice, cmap='hsv', alpha=0.6)
            # Create legend with colored squares
            legend_elements = [
                patches.Patch(color='red', label='ET'),
                patches.Patch(color='blue', label='ED'),
                patches.Patch(color='green', label='NCR')
            ]
            plt.legend(handles=legend_elements, loc='upper right', fontsize=8)
        else:
            plt.imshow(volume_slice, cmap='gray')
        plt.axis('off')
        plt.title(f"{volume_name}")

    plt.tight_layout()
    plt.show()







def visualize_slices(subject, slice_range, seg_cmap='bone', other_cmap='cool'):
    """
    Visualizes a range of slices for all volumes for a patient overlapped together in a single figure.

    Args:
        subject (tio.Subject): A dictionary containing loaded volumes and
            segmentation as torchio.ScalarImage or torchio.LabelMap objects.
        slice_range (tuple): A tuple containing the start and end slices to visualize, e.g., (start_slice, end_slice).
        seg_cmap (str): Colormap for the segmentation volume.
        other_cmap (str): Colormap for all other volumes.
    """

    
    volume_names = list(subject.keys())
    volumes = [subject[name].data.numpy() for name in volume_names]

    num_slices = slice_range[1] - slice_range[0] + 1

    plt.figure(figsize=(12, 4*num_slices))

    for i in range(num_slices):
        plt.subplot(num_slices, len(volume_names), i * len(volume_names) + 1)
        seg_slice = np.squeeze(volumes[0][..., slice_range[0] + i])  
        plt.axis('off')
        plt.title(f"Segmentation - Slice {slice_range[0] + i}")

        for j, volume_data in enumerate(volumes[1:], start=1):
            plt.subplot(num_slices, len(volume_names), i * len(volume_names) + j + 1)
            volume_slice = np.squeeze(volume_data[..., slice_range[0] + i])  
            plt.imshow(volume_slice, cmap=other_cmap)
            plt.axis('off')
            plt.title(f"{volume_names[j]} - Slice {slice_range[0] + i}")

    plt.tight_layout()
    plt.show()



def visualize_overlapped_slices(subject, start_slice=75, end_slice=120, slice_spacing=5, cmap='gray', seg_cmap='hsv'):
    """
    Visualizes a specified range of slices from a patient with all volumes and labels overlaid in a single image 
    using different colormaps for the segmentation volume and other volumes.

    Args:
        subject (tio.Subject): A dictionary containing loaded volumes and
                               segmentation as torchio.ScalarImage or torchio.LabelMap objects.
        start_slice (int, optional): The starting slice number (inclusive). Defaults to 75.
        end_slice (int, optional): The ending slice number (exclusive). Defaults to 120.
        slice_spacing (int, optional): The spacing between plotted slices. Defaults to 5.
        cmap (str, optional): Colormap for non-segmentation volumes. Defaults to 'gray'.
        seg_cmap (str, optional): Colormap for the segmentation volume. Defaults to 'hsv'.
    """
    volumes = {name: subject[name].data.numpy() for name in subject.keys()}

    total_slices = end_slice - start_slice
    num_slices_to_show = min(total_slices, 30)  # Maximum of 30 slices to show
    
    # Calculate the grid size based on the number of slices
    grid_cols = min(num_slices_to_show, 4)  # Increase number of columns
    grid_rows = int(np.ceil(num_slices_to_show / grid_cols))
    
    # Calculate the figure size based on the grid size
    fig_width = 15
    fig_height = fig_width * (grid_rows / grid_cols)
    
    plt.figure(figsize=(fig_width, fig_height))
    plt.subplots_adjust(wspace=0.002, hspace=0.002)  # Reduce spacing between plots
    plt.gcf().patch.set_facecolor('black')  # Set the background color to black

    # Calculate the total number of iterations needed with spacing
    num_iterations = int(np.ceil(total_slices / slice_spacing))

    for i in range(num_iterations):
        # Calculate the slice number for this iteration
        slice_number = start_slice + i * slice_spacing
        if slice_number >= end_slice:
            break

        slice_data = {name: volume[..., slice_number] for name, volume in volumes.items()}

        for name, data in slice_data.items():
            if data.ndim > 2:
                slice_data[name] = np.squeeze(data)

        ax = plt.subplot(grid_rows, grid_cols, i + 1)

        # Overlay all non-segmentation volumes first
        for name, slice in slice_data.items():
            if name != "seg":
                ax.imshow(slice, cmap=cmap, alpha=0.5)

        # Overlay the segmentation volume, masking the background
        if "seg" in slice_data:
            seg_mask = slice_data["seg"] > 0  # Assuming segmentation mask is non-zero in the region of interest
            seg_colored = np.ma.masked_where(~seg_mask, slice_data["seg"])
            ax.imshow(seg_colored, cmap=seg_cmap, alpha=0.5)

        ax.axis('off')

    plt.tight_layout()
    legend_elements = [
                patches.Patch(color='red', label='ET'),
                patches.Patch(color='lightgreen', label='ED'),
                patches.Patch(color='purple', label='NCR')
            ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=8)
    plt.show()


def visualize_sagittal_slices(subject, volume_name, slice_range2, step=1, num_columns=2, cmap='bone'):
    """
    Visualizes sagittal slices of a specific volume for a patient within a custom range with a specified step.

    Args:
        subject (tio.Subject): A dictionary containing loaded volumes and
            segmentation as torchio.ScalarImage or torchio.LabelMap objects.
        volume_name (str): Name of the volume to visualize (e.g., 't1c', 't1n', 't2f', 't2w', 'seg').
        slice_range (tuple): A tuple containing the start and end slices to visualize, e.g., (start_slice, end_slice).
        step (int): The step size to skip slices. Defaults to 1 (no skipping).
        num_columns (int): Number of columns for the subplots grid. Defaults to 2.
        cmap (str): Colormap for the visualization. Defaults to 'gray'.
    """

    volume_data = subject[volume_name].data.numpy()

    # Remove singleton dimensions
    volume_data = np.squeeze(volume_data)

    # Calculate the number of slices to visualize
    num_slices = (slice_range2[1] - slice_range2[0] + 1) // step

    # Calculate the number of rows and columns for the subplots grid
    num_rows = (num_slices + num_columns - 1) // num_columns
    num_plots = num_rows * num_columns

    # Create subplots
    plt.figure(figsize=(6 * num_columns, 3 * num_rows))
    for i, slice_index in enumerate(range(slice_range2[0], slice_range2[1] + 1, step), 1):
        if i <= num_plots:  # Ensure we don't create more subplots than required
            # Selecting the sagittal slice
            sagittal_slice = volume_data[slice_index, :, :]

            # Rotating the slice for visualization (if necessary)
            sagittal_slice_rotated = rotate(sagittal_slice, 90, reshape=True)

            # Plotting the sagittal slice
            plt.subplot(num_rows, num_columns, i)
            plt.imshow(sagittal_slice_rotated, cmap=cmap)
            plt.axis('off')
            plt.title(f"Sagittal Slice {slice_index} - {volume_name}")

    plt.tight_layout()
    plt.show()

