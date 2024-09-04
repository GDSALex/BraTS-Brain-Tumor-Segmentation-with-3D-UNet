

import torchio as tio
import torch

landmarks_dict = {}
volume_types = ["t1c", "t1n", "t2f", "t2w"]

for volume in volume_types:
    landmarks_dict[volume] = torch.load(f'{volume}_landmarks.pt')


# Define transformation pipeline
def transform():
    include = ['t1c', 't1n', 't2f', 't2w']
    transform = tio.Compose([
        tio.ToCanonical(include=include),
        tio.CropOrPad((148, 148, 148), include=include + ['seg']),
        tio.RandomFlip(axes=['LR', 'AP', 'IS'], p=0.2, include=include + ['seg']),
        tio.HistogramStandardization(landmarks_dict, include=include), 
        tio.ZNormalization(include=include),  
        tio.OneOf({
            tio.RandomAffine(include=include + ['seg']): 0.2,
            tio.RandomElasticDeformation(include=include + ['seg']): 0.2,
        }),
        tio.OneOf({
            tio.RandomNoise(include=include): 0.4,
            tio.RandomSpike(include=include):0.1,
        }),
        tio.RandomBiasField(p=0.2, include=include),
        tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.2, include=include),
        
    ])
    return transform


