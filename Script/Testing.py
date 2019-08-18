from Script.Model.whole_model import whole_model
from Script.Manage_dataset.Data_generator import data_gen
from Script import prediction
from keras.layers import Input
import numpy as np
IMPORT CV2

BATCH_SIZE = 4      # Scelto da me abbastanza random, ragionevole
STRIDE = 8
NO_OF_TESTING_IMAGES = len(os.listdir(test_frame_path))

def test(model, stride):

    test_frame_path = '../../Dataset/test_frames'
    test_gen = data_gen(test_frame_path, batch_size = BATCH_SIZE)

    predictions = model.predict_generator( test_gen, steps=(NO_OF_TESTING_IMAGES//BATCH_SIZE) )
    
    num_imgs = predictions.shape[1]
    pose_imgs = []
    for img in range(num_imgs)
        pose = prediction.argmax_predict(predictions, stride)

    ##### DA RICONTROLLARE

    #DA AGGIUNGERE IL MODO DI VISUALIZZARE IL RISULTATO

if __name__ == '__main__':
    model = load_model('Model.h5')
    test(model, STRIDE)
