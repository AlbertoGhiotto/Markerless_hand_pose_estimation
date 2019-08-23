from Script.Model.whole_model import whole_model
from Script.Manage_dataset.Data_generator import data_gen
from Script.Loss import loss
from keras.layers import Input
from keras.optimizers import Adam
from keras.model import load_model
import os

train_frame_path = '../Dataset/train_frames'
val_frame_path = '../Dataset/val_frames'

NO_OF_TRAINING_IMAGES = len(os.listdir(train_frame_path))
NO_OF_VAL_IMAGES = len(os.listdir(val_frame_path))
NO_OF_EPOCHS = 200000     
BATCH_SIZE = 4      


def train(model):

  # Train the model
  train_gen = data_gen(train_frame_path, batch_size = BATCH_SIZE)
  val_gen = data_gen(val_frame_path, batch_size = BATCH_SIZE)


  model.compile( optimizer=Adam(lr=2E-4), loss=weighted_cross_entropy(0.8), metrics=['accuracy'] )
  model.fit_generator( train_gen, epochs=NO_OF_EPOCHS, 
                            steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                            validation_data=val_gen, 
                            validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE) )

  model.save("drive/My Drive/Colab Notebooks/Model/Model.h5")
  print("Saved model to drive")

try:
  model = load_model("drive/My Drive/Colab Notebooks/Model/Model.h5", custom_objects={'loss': weighted_cross_entropy(0.8)})
  print("Model loaded from Drive")
except IOError:
  try:
    model.load_weights('./Checkpoints/my_checkpoint')
  except IOError:
    model = whole_model()
train(model)
