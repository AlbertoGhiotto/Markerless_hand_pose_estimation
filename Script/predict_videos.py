import cv2
from keras.preprocessing import image as img
from keras.models import load_model
from Script.prediction import visualize
import numpy as np

STRIDE = 8


def video_prediction(video, model, stride, show_result = True):
  # dimensions of our images
  #img_width, img_height = 512, 512

  vidcap = cv2.VideoCapture('group_project/test.mp4')
  
  while True:      
    grabbed,image = vidcap.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
      break

    # predicting images
    image = img.img_to_array(image)
    image = np.expand_dims(image, axis=3)
    
    predictions = model.predict(image, batch_size=1, verbose=1, )
    
    num_joints = predictions.shape[3]

    pose = []
    for joint_idx in range(num_joints):
      maxloc = np.unravel_index(np.argmax(predictions[:, :, joint_idx]), predictions[:, :, joint_idx].shape)
      pos_f8 = (np.array(maxloc).astype('float') * stride + 0.5 * stride)
      pose.append(np.hstack((pos_f8[::-1], [predictions[maxloc][joint_idx]])))
    pose = np.array(pose)

    # Plot the pose on the original image
    if show_result: 
      visualize(image, pose)


if __name__ == "__main__":
    video_path = 'group_project/test.mp4'
    model = load_model('/content/drive/My Drive/Colab Notebooks/Model/Model.h5')
    video_prediction(video_path, model, STRIDE)