import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os
import scipy.linalg

# Define a simple dataset loading class
class SimpleDataset(Dataset):
    def __init__(self, folder):
        self.files = [os.path.join(folder, file) for file in os.listdir(folder)]
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert('RGB')
        return self.transform(image)

# Calculate feature statistics: mean and covariance
def calculate_statistics(dataloader):
    inception = models.inception_v3(pretrained=True, transform_input=False)
    inception.fc = torch.nn.Identity()
    inception.eval()

    # Move the model to CUDA if available
    if torch.cuda.is_available():
        inception = inception.cuda()

    features = []
    with torch.no_grad():
        for batch in dataloader:
            # Similarly, ensure input data is on the same device
            if torch.cuda.is_available():
                batch = batch.cuda()
            feature = inception(batch)
            features.append(feature.cpu().numpy())  # Finally move the features back to CPU for further processing
    features = np.concatenate(features, axis=0)
    mean = np.mean(features, axis=0)
    cov = np.cov(features, rowvar=False)
    return mean, cov

# Calculate FID score
def calculate_fid(mean1, cov1, mean2, cov2):
    diff = mean1 - mean2
    covmean = scipy.linalg.sqrtm(cov1.dot(cov2), disp=False)[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(cov1 + cov2 - 2 * covmean)
    return fid

# Load datasets
label_list=['car','clothes','dog']
label=label_list[2]
folderA = f'test_data/image/{label}'  # Path to real image folder
folderB = f'sampled_data/repaint_n10_mixed/{label}'  # Path to generated image folder

datasetA = SimpleDataset(folderA)
datasetB = SimpleDataset(folderB)

dataloaderA = DataLoader(datasetA, batch_size=64, shuffle=False)
dataloaderB = DataLoader(datasetB, batch_size=64, shuffle=False)

# Calculate statistics
meanA, covA = calculate_statistics(dataloaderA)
meanB, covB = calculate_statistics(dataloaderB)

# Calculate FID score
fid_score = calculate_fid(meanA, covA, meanB, covB)
print(f'{label} FID score: {fid_score}')
