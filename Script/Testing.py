from Script.Model.whole_model import whole_model
from Script.Manage_dataset.Data_generator import data_gen
from keras.layers import Input
import numpy as np


##### DA MODIFICARE:  l'istanza del model deve eseere coerente con quella di training
input_tensor = Input(shape=(None, None, 3))

model = whole_model(input_tensor)

BATCH_SIZE = 4      # Scelto da me abbastanza random, ragionevole
NO_OF_TESTING_IMAGES = len(os.listdir(test_frame_path))

test_frame_path = 'https://raw.githubusercontent.com/AlbertoGhiotto/group_project/master/Dataset/test_frames'
test_gen = data_gen(test_frame_path, batch_size = BATCH_SIZE)

predictions = model.predict_generator( test_gen, steps=(NO_OF_TESTING_IMAGES//BATCH_SIZE) )

num_joints = predictions.shape[2]
pose = []
for joint_idx in range(num_joints):
    maxloc = np.unravel_index(np.argmax(predictions[:, :, joint_idx]), predictions[:, :, joint_idx].shape)
    pos_f8 = (np.array(maxloc).astype('float') * stride + 0.5 * stride)
    pose.append(np.hstack((pos_f8[::-1], [predictions[maxloc][joint_idx]])))
pose = np.array(pose)

##### DA RICONTROLLARE

#DA AGGIUNGERE IL MODO DI VISUALIZZARE IL RISULTATO
