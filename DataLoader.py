import os
import torchio as tio
from tqdm import tqdm 
import torch



    
class NiftiDataset(tio.data.SubjectsDataset):

    """
    Dataset Class to load the NIFTI data of the 2023 BraTS Challenge.

    Args:
        root_dir (str): directory of training data.
        volumes (str): volumes of which the CT scan is composed; T1-weighted with
        contrast enhancement (t1c), T1-weighted without contrast enhancement
        (t1n), T2-weighted fluid-attenuated inversion recovery (t2f) and
        T2-weighted (t2w).
        seg_volume (str): volume of labels
        transform: whether it performs transformations or not. It defaults to None. 
    """


    def __init__(self, root_dir, volumes=["t1c", "t1n", "t2f", "t2w"], seg_volume="seg", transform = True):
        self.root_dir = root_dir
        self.volumes = volumes
        self.seg_volume = seg_volume
        self.subjects = self.load_volumes()
        self.transform = transform

    def __len__(self):
        """
        Returns the number of patients (samples) in the dataset.
        """
        return len(self.subjects)    


    def load_patient(self, patient_folder):
        """
        Loads NIFTI files for each volume and segmentation volume for a patient.
        """
        subject = {}
        for volume in self.volumes:
            nifti_path = os.path.join(self.root_dir, patient_folder, f"{patient_folder}-{volume}.nii.gz")
            if os.path.exists(nifti_path):
                subject[volume] = tio.ScalarImage(nifti_path)
            else:
                print(f"File not found: {nifti_path}")
        seg_path = os.path.join(self.root_dir, patient_folder, f"{patient_folder}-{self.seg_volume}.nii.gz")
        if os.path.exists(seg_path):
            subject[self.seg_volume] = tio.LabelMap(seg_path)
            
        else:
            print(f"Segmentation file not found: {seg_path}")

        return subject

    def load_volumes(self):
        """
        Loads NIFTI files for all patients in the dataset.
        
        """
        subjects = []
        for patient_folder in tqdm(os.listdir(self.root_dir)):
            if os.path.isdir(os.path.join(self.root_dir, patient_folder)):
                subject = self.load_patient(patient_folder)
                subjects.append(subject)
        return subjects
    

    def __getitem__(self, index) -> tio.Subject:
        subject = self.subjects[index]
        for key, value in subject.items():
            if isinstance(value, tio.ScalarImage):
                subject[key] = value.tensor  
            if isinstance(value, tio.LabelMap):
                seg_tensor = value.tensor
                #print(f"Unique labels before adjustment: {torch.unique(seg_tensor)}")
                seg_tensor = torch.clamp(seg_tensor, min=0, max=3)  # Ensure labels are within the valid range
                #print(f"Unique labels after adjustment: {torch.unique(seg_tensor)}")
                subject[key] = seg_tensor  # No need to modify the labels

        if self.transform:
            subject = self.transform(subject)
        return subject
