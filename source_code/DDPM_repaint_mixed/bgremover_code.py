from skimage import io
import torch
import os
from PIL import Image
from briarmbg import BriaRMBG
from utilities import preprocess_image, postprocess_image

# Define paths
model_path = 'model.pth'
input_folder = 'input_images'  # This is the input folder (original images)
output_folder = 'result_rb'  # This is the folder to save the result (background removed)

# Load the model
net = BriaRMBG()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net.load_state_dict(torch.load(model_path, map_location=device))
net.to(device)
net.eval()

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate over images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        # Load the original image
        im_path = os.path.join(input_folder, filename)
        orig_im = io.imread(im_path)
        orig_im_size = orig_im.shape[0:2]

        # Preprocess the image
        model_input_size = [orig_im.shape[0], orig_im.shape[1]]  # Adjust model input size
        image = preprocess_image(orig_im, model_input_size).to(device)

        # Process the image
        result = net(image)

        # Postprocess the image
        result_image = postprocess_image(result[0][0], orig_im_size)
        pil_im = Image.fromarray(result_image)

        # Create a new image with transparent background
        no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
        orig_image = Image.open(im_path)
        no_bg_image.paste(orig_image, mask=pil_im)

        # Save the result
        output_path = os.path.join(output_folder, filename.replace('.', '_bg_removed.'))
        # Convert RGBA image to RGB before saving
        no_bg_image.convert("RGB").save(output_path)

print("Background removal process completed.")
