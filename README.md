# Sphere Center Detection in PyTorch - CircleNet
Simple working example of a convolutional neural net implemented in PyTorch that predicts the center of a circle in a black and white image.

## Overview
The project is organized into the following key components:

***Synthetic Data Generation:*** Involves generating black and white images with circles of random sizes and positions, and storing the coordinates of their centers.

***CircleNet Model:*** A CNN model implemented in PyTorch. The model takes an image as input and outputs the coordinates of the circle's center.

***Training Procedure:*** The model is trained on the synthetic dataset with the aim to minimize the discrepancy between the model's predicted circle center and the true circle center.

***Prediction & Visualization:*** The prediction script loads a trained CircleNet model and performs predictions on a new set of images. Predicted circle centers and true circle centers are visualized for comparison.
## Installation
```bash
git clone XXX
cd circlenet
pip3 install -r requirements.txt
```

## Image Generation
The script "generate_synthetic_data.py" can be used to generate a synthetic dataset of black and white images, each containing a circle of random size and position. Along with each generated image, the true coordinates of the circle's center are stored as a label. 
```
python3 generate_synthetic_spheres.py
```

## Model training
```
python3 train.py
```

## Visualization
```
python3 visualize_results.py
```