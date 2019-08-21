from Script.Model.whole_model import whole_model
from Script.Manage_dataset.Data_generator import data_gen
from Script import prediction
from Script.prediction import visualize
from keras.layers import Input
import numpy as np
IMPORT CV2

BATCH_SIZE = 4     
STRIDE = 8
NO_OF_TESTING_IMAGES = len(os.listdir(test_frame_path))
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512



def test(model, stride, show_result = True):

    test_frame_path = '../Dataset/test_frames'
    test_gen = data_gen(test_frame_path, batch_size = BATCH_SIZE, shuffle=False)

    predictions = model.predict_generator( test_gen, steps=(NO_OF_TESTING_IMAGES//BATCH_SIZE) )
    
    n = os.listdir(test_frame_path)
    num_imgs = len(n)
    num_joints = predictions.shape[3]
    pose_imgs = []

    for img in range(num_imgs):
        pose = prediction.argmax_predict(predictions, stride)
        if show_result:
            image = cv2.imread(test_frame_path+'/'+n[img])    
            image =  cv2.resize(image, (DEFAULT_HEIGHT, DEFAULT_WIDTH))
            visualize(image, pose)
        pose_imgs.append(pose)
    
   

if __name__ == '__main__':
    model = load_model('Model/Model.h5')
    test(model, STRIDE)
