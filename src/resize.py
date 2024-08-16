from PIL import Image
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000
import os

# Resize png file in png directory
for filename in os.listdir("png/"):
    if filename.endswith(".png"):
        png_file = os.path.join("png", filename)
        img = Image.open(png_file)
        # Resize the image to 256x256
        img_resized = img.resize((256, 256), PIL.Image.Resampling.LANCZOS)
        # # Save the resized image
        dest_dir = os.path.join("png", "resized", filename)
        img_resized.save(dest_dir)
