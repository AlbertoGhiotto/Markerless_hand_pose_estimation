import cv2
from keras.preprocessing import image as img
from keras.models import load_model
from Script.prediction import visualize
from Script import prediction
from Script.Loss import weighted_cross_entropy
import numpy as np

STRIDE = 8


def video_prediction(video, model, stride, show_result = True):
  # dimensions of our images
  #img_width, img_height = 512, 512

  vidcap = cv2.VideoCapture(video)
  
  while True:      
    grabbed,image = vidcap.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
      break

    # predicting images
    image_array = img.img_to_array(image)
    image_array = np.expand_dims(image, axis=0)
    
    predictions = model.predict(image_array, batch_size=1, verbose=1, )

    pose = prediction.argmax_predict(predictions, stride)

    # Plot the pose on the original image
    if show_result: 
      visualize(image, pose)


if __name__ == "__main__":
    video_path = 'group_project/test.mp4'
    model = load_model('Model/Model.h5', custom_objects={'loss': weighted_cross_entropy(0.8)})
    video_prediction(video_path, model, STRIDE)
