from PIL import Image
import numpy as np
import os
from sklearn.preprocessing import MultiLabelBinarizer

# Directory where your images are saved
image_dir = 'images'

# The size to which to resize your images. This should be smaller than or equal to the size of your images.
image_size = (512, 512)

# Get a list of all image file paths
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

# Preallocate a numpy array for your images
images = np.empty((len(image_files), image_size[0], image_size[1], 3))

# Initialize an empty list for the labels
labels = []

# Initialize the multilabel binarizer
mlb = MultiLabelBinarizer()

# Loop over all image files
for i, image_file in enumerate(image_files):
    # Load image
    image = Image.open(image_file)
    # Resize image
    image = image.resize(image_size)
    # Convert image to numpy array and normalize pixel values to be between 0 and 1
    image = np.array(image) / 255.0
    # Add image to the array
    images[i] = image
    
    # Extract font name, style and character from filename
    base_name = os.path.basename(image_file)
    font_name, font_style, char = base_name.split('-')[0], base_name.split('-')[1], base_name.split('-')[2].split('.')[0]
    labels.append((font_name, font_style, char))

    # Print the filename and the extracted label
    print(f"Filename: {image_file}, Extracted label: {(font_name, font_style, char)}")

# Convert labels to binary values
labels = mlb.fit_transform(labels)

# Save the preprocessed images and labels as numpy files
np.save('images.npy', images)
np.save('labels.npy', labels)