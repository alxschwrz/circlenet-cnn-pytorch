import numpy as np
import cv2
import os
import pandas as pd
from numpy.random import default_rng

if not os.path.exists('images'):
    os.makedirs('images')

rng = default_rng()

# Define variables
num_images = 200
image_size = 96
radius_low = 10
radius_high = 20
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
