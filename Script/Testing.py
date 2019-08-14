from Script.Model.whole_model import whole_model
from Script.Manage_dataset.Data_generator import data_gen
from keras.layers import Input


##### DA MODIFICARE:  l'istanza del model deve eseere coerente con quella di training
input_tensor = Input(shape=(None, None, 3))

model = whole_model(input_tensor)

BATCH_SIZE = 4      # Scelto da me abbastanza random, ragionevole
NO_OF_TESTING_IMAGES = len(os.listdir(test_frame_path))

test_frame_path = 'https://raw.githubusercontent.com/AlbertoGhiotto/group_project/master/Dataset/test_frames'
test_gen = data_gen(test_frame_path, batch_size = BATCH_SIZE)

model.predict_generator( test_gen, steps=(NO_OF_TESTING_IMAGES//BATCH_SIZE) )

#DA AGGIUNGERE IL MODO DI VISUALIZZARE IL RISULTATO
