from Script.Model.whole_model import whole_model
from Script.Manage_dataset.Data_generator import data_gen
from keras.layers import Input
from keras.optimizers import Adam

train_frame_path = 'https://raw.githubusercontent.com/AlbertoGhiotto/group_project/master/Dataset/train_frames'
val_frame_path = 'https://raw.githubusercontent.com/AlbertoGhiotto/group_project/master/Dataset/val_frames'

NO_OF_TRAINING_IMAGES = len(os.listdir(train_frame_path))
NO_OF_VAL_IMAGES = len(os.listdir(val_frame_path))
NO_OF_EPOCHS = 50   # Scelto da me abbastanza random, se non sbaglio nell'articolo parlava di valori molto più elevati
BATCH_SIZE = 4      # Scelto da me abbastanza random, ragionevole

#height = 224 #dimensions of image
#width = 224
#channel = 3

#input_tensor = Input(shape=(224, 224, 3))
input_tensor = Input(shape=(None, None, 3))

model = whole_model(input_tensor)


# Train the model
train_gen = data_gen(train_frame_path, batch_size = BATCH_SIZE)
val_gen = data_gen(val_frame_path, batch_size = BATCH_SIZE)


model.compile( optimizer=Adam(lr=1E-5), loss='categorical_crossentropy', metrics=['accuracy'] )#, loss_weights=[1., 0.2])
model.fit_generator( train_gen, epochs=NO_OF_EPOCHS, 
                          steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                          validation_data=val_gen, 
                          validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE) )

model.save('Model.h5')
