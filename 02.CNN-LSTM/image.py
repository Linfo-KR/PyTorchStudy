import os

from PIL import Image

from dataset import *

def reshape_image(image, shape):
    """
    Resize an image to the given shape.
    """
    return image.resize(shape, Image.Resampling.LANCZOS)

def reshape_images(image_dir, output_dir, shape):
    """
    Reshape the images in 'image_dir' and save into 'output_dir'.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    images = os.listdir(image_dir)
    num_img = len(images)
    
    for i, img in enumerate(images):
        with open(os.path.join(image_dir, img), 'r+b') as f:
            with Image.open(f) as image:
                image = reshape_image(image, shape)
                image.save(os.path.join(output_dir, img), image.format)
        
        if (i+1) % 100 == 0:
            print("[{}/{}] Resized the images and saved into '{}'.".format(i+1, num_img, output_dir))
            

if __name__ == '__main__':
    image_dir =  storage_dir + 'train2017/'
    output_dir = storage_dir + 'resized/'
    image_shape = [256, 256]
    reshape_images(image_dir, output_dir, image_shape)