import torch
from torchvision import models, transforms

from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os

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

# Calculate Inception Score
def inception_score(loader, model, splits=10):
    preds = []
    for batch in loader:
        if torch.cuda.is_available():
            batch = batch.cuda()
        with torch.no_grad():
            pred = torch.nn.functional.softmax(model(batch), dim=1)  # Use softmax to get probability distribution
        preds.append(pred.cpu().numpy())

    preds = np.concatenate(preds, 0) + 1e-16  # Avoid cases where probabilities are 0

    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

# Load pre-trained Inception model
inception_model = models.inception_v3(pretrained=True, transform_input=False)
inception_model.eval()
if torch.cuda.is_available():
    inception_model.cuda()

# Load generated fake images
label_list=['car','clothes','dog']
label=label_list[2]
folderF = f'sampled_data/repaint_n10_mixed/{label}'   # Path to the folder containing fake images
datasetF = SimpleDataset(folderF)
fake_loader = DataLoader(datasetF, batch_size=64, shuffle=True)
# Use the defined function to compute the Inception Score
score, std = inception_score(fake_loader, inception_model)
print(f"{label} Inception Score: {score}, Std: {std}")
