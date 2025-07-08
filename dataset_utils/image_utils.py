import os
from pathlib import Path
from PIL import Image
import logging

def resize_image(image, target_width, target_height):
    """
    Resize a PIL image to the target dimensions while preserving aspect ratio.
    
    Args:
        image (PIL.Image): The image to resize
        target_width (int): The target width
        target_height (int): The target height
        
    Returns:
        PIL.Image: The resized image
    """
    if image.width == target_width and image.height == target_height:
        return image
    
    # Calculate aspect ratios
    aspect_ratio = image.width / image.height
    target_aspect_ratio = target_width / target_height
    
    # Determine dimensions for resizing
    if aspect_ratio > target_aspect_ratio:
        # Image is wider than target
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        # Image is taller than target
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    
    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create a new blank image with the target dimensions
    new_image = Image.new("RGB", (target_width, target_height))
    
    # Calculate position to paste the resized image (center it)
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    
    # Paste the resized image onto the blank image
    new_image.paste(resized_image, (paste_x, paste_y))
    
    return new_image

def resize_images_in_directory(directory_path, target_width=None, target_height=None):
    """
    Resize all images in a directory to the target dimensions.
    
    Args:
        directory_path (str or Path): Path to the directory containing images
        target_width (int, optional): Target width. If None, uses the width from the first image.
        target_height (int, optional): Target height. If None, uses the height from the first image.
    """
    directory_path = Path(directory_path)
    
    # Get all image files in the directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = [f for f in directory_path.iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        logging.warning(f"No image files found in {directory_path}")
        return
    
    # If target dimensions not provided, use the first image's dimensions
    if target_width is None or target_height is None:
        first_image = Image.open(image_files[0])
        target_width = target_width or first_image.width
        target_height = target_height or first_image.height
        first_image.close()
    
    # Process each image
    total_images = len(image_files)
    for i, image_path in enumerate(image_files, 1):
        try:
            with Image.open(image_path) as img:
                # Skip if already the right size
                if img.width == target_width and img.height == target_height:
                    continue
                
                # Resize and save
                resized_img = resize_image(img, target_width, target_height)
                resized_img.save(image_path)
                
            if i % 100 == 0 or i == total_images:
                logging.info(f"Resized {i}/{total_images} images")
                
        except Exception as e:
            logging.error(f"Error resizing {image_path}: {e}")