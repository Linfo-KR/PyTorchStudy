import os
import json
import h5py

import numpy as np

from PIL import Image
from pycocotools.coco import COCO

from dataset import *

def reshape_image(image, shape):
    """
    Resize an image to the given shape and convert it to uint8 format.
    """
    # Resize the image
    resized_image = image.resize(shape, Image.Resampling.LANCZOS)
    # Convert the image to RGB format
    rgb_image = resized_image.convert('RGB')
    # Convert the image to numpy array
    img_array = np.array(rgb_image)
    return img_array

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
            
def create_hdf5_dataset(image_dir, coco_json_dir, output_file, shape):
    """
    Create HDF5 dataset from images and COCO annotations.
    """
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    
    coco_data = COCO(coco_json_dir)
    image_ids = list(coco_data.imgs.keys())
    
    with h5py.File(output_file, 'w') as hdf5_file:
        img_dataset = hdf5_file.create_dataset('images', (len(image_ids), *shape, 3), dtype=np.uint8)
        captions_dataset = hdf5_file.create_dataset('captions', (len(image_ids),), dtype=h5py.special_dtype(vlen=str))
        
        for i, img_id in enumerate(image_ids):
            img_info = coco_data.loadImgs(img_id)[0]
            img_file = os.path.join(image_dir, img_info['file_name'])
            
            with open(img_file, 'r+b') as f:
                with Image.open(f) as image:
                    image = reshape_image(image, shape)
                    img_array = np.array(image)
                    img_dataset[i] = img_array
            
            ann_ids = coco_data.getAnnIds(imgIds=img_id)
            anns = coco_data.loadAnns(ann_ids)
            captions = ' '.join([ann['caption'] for ann in anns])
            captions_dataset[i] = captions
            
            if (i+1) % 100 == 0:
                print("[{}/{}] Processed images and captions.".format(i+1, len(image_ids)))
            

if __name__ == '__main__':
    image_dir = os.path.join(storage_dir, 'train2017/')
    anns_train_dir = os.path.join(storage_dir, 'annotations/captions_train2017.json')
    # output_dir = os.path.join(storage_dir, 'resized/')
    output_dir = os.path.join(storage_dir, 'hdf5/')
    output_file = os.path.join(output_dir, 'CustomCOCO.hdf5')
    image_shape = [256, 256]
    create_hdf5_dataset(image_dir, anns_train_dir, output_file, image_shape)