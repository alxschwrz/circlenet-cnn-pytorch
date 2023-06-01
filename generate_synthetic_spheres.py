import numpy as np
import cv2
import os
import pandas as pd
from numpy.random import default_rng
import argparse

if not os.path.exists('images'):
    os.makedirs('images')

rng = default_rng()

parser = argparse.ArgumentParser()
parser.add_argument('--num_images', type=int, default=100, help='Number of images')
parser.add_argument('--image_size', type=int, default=96, help='Size of the images')
parser.add_argument('--radius_low', type=int, default=10, help='Minimum radius of the spheres')
parser.add_argument('--radius_high', type=int, default=20, help='Maximum radius of the spheres')
args = parser.parse_args()

# Define variables
num_images = args.num_images
image_size = args.image_size
radius_low = args.radius_low
radius_high = args.radius_high
center_loc = image_size // 2
center_scale = image_size // 4

labels = []

for i in range(num_images):

    img = np.zeros((image_size, image_size), np.uint8)

    # Adjust sphere center generation to use a Gaussian distribution centered around the image origin
    u = int(rng.normal(loc=center_loc, scale=center_scale))
    v = int(rng.normal(loc=center_loc, scale=center_scale))

    # Clip the values to avoid the sphere going out of the image frame
    u = np.clip(u, radius_high, image_size - radius_high)
    v = np.clip(v, radius_high, image_size - radius_high)

    # Make sure the radius is within the limit so the whole sphere is within the image frame
    r = rng.integers(radius_low, min(u, v, image_size-u, image_size-v))


    cv2.circle(img, (u, v), r, (255), -1)
    noise = rng.normal(0.0, 0.8, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)

    # Apply Gaussian blur for smoothness
    img = cv2.GaussianBlur(img, (5, 5), 50)

    img_path = f'images/img_{i}.png'
    cv2.imwrite(img_path, img)

    labels.append([img_path, u, v, r])

labels_df = pd.DataFrame(labels, columns=['image_path', 'center_u', 'center_v', 'radius'])
labels_df.to_csv('labels.csv', index=False)
