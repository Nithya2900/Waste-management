import os
import shutil
import random

# Set paths
dataset_dir = 'C:/Users/hp15s/Desktop/vit/dataset'
train_dir = 'C:/Users/hp15s/Desktop/vit/train'
test_dir = 'C:/Users/hp15s/Desktop/vit/test'

# Create directories for train and test sets
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Define the classes
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Split ratio
train_ratio = 0.8

for class_name in class_names:
    # Get the full path for each class
    class_path = os.path.join(dataset_dir, class_name)
    images = os.listdir(class_path)
    
    # Shuffle images for random splitting
    random.shuffle(images)
    
    # Calculate split index
    split_index = int(len(images) * train_ratio)
    
    # Create subfolders for each class in train and test directories
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
    
    # Move the images into train and test directories
    for i, image in enumerate(images):
        src_path = os.path.join(class_path, image)
        if i < split_index:
            dst_path = os.path.join(train_dir, class_name, image)
        else:
            dst_path = os.path.join(test_dir, class_name, image)
        shutil.copy(src_path, dst_path)
