import numpy as np

def read_csv_labels():
  file = 'Dataset/dataset.csv'
  labels = np.loadtxt(file, delimiter=',')
  X_labels = labels[:, 0:15]
  Y_labels = labels[:, 16:31]
  return X_labels, Y_labels

####### DA MOIFICARE PER CREARE IL NOSTRO DATA_GENERATOR CHE LEGGE I VALORI DAL CSV PER OGNI IMMAGINE E DENTRO QUESTA FUNZIONE NE CREA LA MASCHERA, BISOGNA AGGIUNGERCI IL FATTO CHE LA SIZE DELL'IMAGINE LA RICEVA IN INPUT 
####### E BISOGNA STARE ATTENTI ALLA DIMENSIONE DELLA MASK IN OUTPUT    reference(https://towardsdatascience.com/a-keras-pipeline-for-image-segmentation-part-1-6515a421157d)

import cv2

STRIDE = 8
EPSILON = 8     ## Questo valore poi andrebbe un po' aggiustato
NUM_JOINTS = 16

def create_mask(x,y,epsilon):                   ############# Questo poi forse sarebbe meglio metterlo come circonferenza intorno al punto invece di un quadrato
  mask = np.zeros((512/STRIDE, 512/STRIDE, NUM_JOINTS)).astype('float')
  for joint in NUM_JOINTS
    for i in range(-epsilon/STRIDE, epsilon/STRIDE):
      for j in range(-epsilon/STRIDE, epsilon/STRIDE):
        mask[x[joint] + i, y[joint] + j] = 1
  return mask

def data_gen(img_folder, batch_size):
  c = 0
  n = os.listdir(img_folder) #List of training images
  random.shuffle(n)
  X_labels, Y_labels = read_csv_labels()
  
  while (True):         ######## QUESTO WHILE TRUE FORSE ANDREBBE CAMBIATO
    img = np.zeros((batch_size, 512, 512, 3)).astype('float')
    mask = np.zeros((batch_size, 512/STRIDE, 512/STRIDE, NUM_JOINTS)).astype('float')

    for i in range(c, c+batch_size): #initially from 0 to 16, c = 0. 

      train_img = cv2.imread(img_folder+'/'+n[i])/255.
      train_img =  cv2.resize(train_img, (512, 512))# Read an image from folder and resize
      
      img[i-c] = train_img #add to array - img[0], img[1], and so on.        ############# NON SO PERCHÈ CI ABBIANO MESSO IL SEGNO - INVECE DI +
                                                   
      
      id_img =  #bisogna estrarre il numero dal nome del file
      x = X_labels[id_img,:]
      y = Y_labels[id_img,:]
      train_mask = create_mask(x,y,EPSILON)
      train_mask = cv2.resize(train_mask, (512/STRIDE, 512/STRIDE, NUM_JOINTS))
      #train_mask = train_mask.reshape(512, 512, 1) # Add extra dimension for parity with train_img size [512 * 512 * 3]

      mask[i-c] = train_mask

    c+=batch_size
    if(c+batch_size>=len(os.listdir(img_folder))):
      c=0
      random.shuffle(n)
                  # print "randomizing again"
    yield img, mask




train_frame_path = '/path/to/training_frames'
#train_mask_path = '/path/to/training_masks'

val_frame_path = '/path/to/validation_frames'
#val_mask_path = '/path/to/validation_frames'

# Train the model
train_gen = data_gen(train_frame_path, batch_size = 4)
val_gen = data_gen(val_frame_path, batch_size = 4)