from PIL import Image
from pathlib import Path

image_directory = Path("test_selected/image")
sketch_directory = Path("test_selected/sketch")
out_directory = Path("test_selected/combined")

for subdir in image_directory.iterdir():
    (out_directory / subdir.name).mkdir(parents=True, exist_ok=True)
    for image_path in subdir.iterdir():
        sketch_path = sketch_directory / subdir.name / (image_path.stem + ".png")
        out_path = out_directory / subdir.name / (image_path.stem + ".png")

        # Load the images
        image1 = Image.open(sketch_path)
        image2 = Image.open(image_path)

        # Get the dimensions of the images
        width1, height1 = image1.size
        width2, height2 = image2.size

        # Create a new image with the combined width and the height of the tallest image
        new_width = width1 + width2
        new_height = max(height1, height2)
        new_image = Image.new("RGB", (new_width, new_height))

        # Paste the two images onto the new image
        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (width1, 0))

        # Save the new image
        new_image.save(out_path)
