import os
import random
import re
from PIL import Image

DATA_PATH = '../../master/Dataset'
FRAME_PATH = DATA_PATH+'/Imgs'

# Function to split the dataset in training validation and test set

def dataset_split(): 

  # Create folders to hold images and masks
  folders = ['train_frames', 'val_frames', 'test_frames']

  for folder in folders:
    os.makedirs(DATA_PATH + folder)

  # Get all frames, sort them, shuffle them to generate data sets.
  all_frames = os.listdir(FRAME_PATH)

  random.seed(230)
  random.shuffle(all_frames)

  # Generate train, val, and test sets for frames
  train_split = int(0.7*len(all_frames))
  val_split = int(0.9 * len(all_frames))

  train_frames = all_frames[:train_split]
  val_frames = all_frames[train_split:val_split]
  test_frames = all_frames[val_split:]

  def add_frames(dir_name, image):

    img = Image.open(FRAME_PATH+image)
    img.save(DATA_PATH+'/{}'.format(dir_name)+'/'+image)


  frame_folders = [(train_frames, 'train_frames'), (val_frames, 'val_frames'), 
                   (test_frames, 'test_frames')]

  # Add frames

  for folder in frame_folders:

    array = folder[0]
    name = [folder[1]] * len(array)

    list(map(add_frames, name, array))
