import torch
import lpips
import os
import random
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize

# Function to load and resize images from a folder
def load_and_resize_images_from_folder(folder, max_images=None, size=(64, 64)):
    images = []
    filenames = os.listdir(folder)
    if max_images is not None:
        filenames = random.sample(filenames, min(max_images, len(filenames)))
    for filename in filenames:
        img = Image.open(os.path.join(folder, filename)).convert('RGB')
        img = img.resize(size, Image.ANTIALIAS)
        images.append(img)
    return images

# Function to load images from a folder
def load_images_from_folder(folder, max_images=None):
    images = []
    filenames = os.listdir(folder)
    if max_images is not None:
        filenames = random.sample(filenames, max_images)
    for filename in filenames:
        img = Image.open(os.path.join(folder, filename)).convert('RGB')
        images.append(img)
    return images

# Function to calculate LPIPS distance
def calculate_lpips_distance(real_images_folder, fake_images_folder, image_size=(64, 64)):
    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net='alex')
    if torch.cuda.is_available():
        lpips_model = lpips_model.cuda()

    # Determine the number of images in each folder and choose the smaller number
    num_real_images = len(
        [name for name in os.listdir(real_images_folder) if os.path.isfile(os.path.join(real_images_folder, name))])
    num_fake_images = len(
        [name for name in os.listdir(fake_images_folder) if os.path.isfile(os.path.join(fake_images_folder, name))])
    min_images = min(num_real_images, num_fake_images)

    # Load and resize images
    real_images = load_and_resize_images_from_folder(real_images_folder, max_images=min_images, size=image_size)
    fake_images = load_and_resize_images_from_folder(fake_images_folder, max_images=min_images, size=image_size)

    # Convert to tensors and stack
    to_tensor = ToTensor()
    real_images_tensor = torch.stack([to_tensor(image) for image in real_images])
    fake_images_tensor = torch.stack([to_tensor(image) for image in fake_images])

    # Move to GPU if available
    if torch.cuda.is_available():
        real_images_tensor = real_images_tensor.cuda()
        fake_images_tensor = fake_images_tensor.cuda()

    # Calculate LPIPS distance
    with torch.no_grad():
        lpips_distances = lpips_model(real_images_tensor, fake_images_tensor)

    # Return average distance
    return lpips_distances.mean().item()

# Example usage
label_list=['car','clothes','dog']
label=label_list[2]
real_images_folder = f'test_data/image/{label}'  # Path to real images folder
fake_images_folder = f'sampled_data/repaint_n10_mixed/{label}'  # Path to generated images folder
average_distance = calculate_lpips_distance(real_images_folder, fake_images_folder)
print(f"{label} Average LPIPS distance: {average_distance}")
