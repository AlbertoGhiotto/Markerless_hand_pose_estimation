import cv2
import numpy as np

STRIDE = 8
EPSILON = 8     ## Questo valore poi andrebbe un po' aggiustato
NUM_JOINTS = 16
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
ACTUAL_WIDTH = 640
ACTUAL_HEIGHT = 480

def read_csv_labels():
  file = '../../Dataset/dataset.csv'
  labels = np.loadtxt(file, delimiter=',')
  X_labels = labels[:, 0:16]
  Y_labels = labels[:, 16:32]
  return X_labels, Y_labels


def create_mask(x,y,epsilon):                   ############# Questo poi forse sarebbe meglio metterlo come circonferenza intorno al punto invece di un quadrato e magari mettere la gaussiana
  mask = np.zeros((ACTUAL_HEIGHT//STRIDE, ACTUAL_WIDTH//STRIDE, NUM_JOINTS)).astype('float')    
  for joint in range(NUM_JOINTS):
    for i in range(-epsilon//STRIDE, epsilon//STRIDE):
      for j in range(-epsilon//STRIDE, epsilon//STRIDE):
        if (x[joint]//STRIDE + i)<(ACTUAL_HEIGHT//STRIDE) and (y[joint]//STRIDE + j)<(ACTUAL_WIDTH//STRIDE): 
          if x[joint] != -1 and y[joint] != -1:       # If a joint is not visible in the frame
            mask[(x[joint]//STRIDE + i), (y[joint]//STRIDE + j), joint] = 1
  return mask

def data_gen(img_folder, batch_size, shuffle=True):
  c = 0
  n = os.listdir(img_folder) #List of training images
  random.shuffle(n)
  X_labels, Y_labels = read_csv_labels()
  
  while (True):         
    img = np.zeros((batch_size, DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)).astype('float')
    mask = np.zeros((batch_size, DEFAULT_HEIGHT//STRIDE, DEFAULT_WIDTH//STRIDE, NUM_JOINTS)).astype('float')

    for i in range(c, c+batch_size): #initially from 0 to 16, c = 0. 

      train_img = cv2.imread(img_folder+'/'+n[i])/255.
      train_img =  cv2.resize(train_img, (DEFAULT_HEIGHT, DEFAULT_WIDTH))# Read an image from folder and resize
      
      img[i-c] = train_img #add to array - img[0], img[1], and so on.       
                                                   
      # extract the number of the image from the string name
      if str.isdigit(n[i][6]): #takes the number in brackets
        if str.isdigit(n[i][7]): #number with three digits
          img_num = n[i][5] + n[i][6] + n[i][7]
        else: #number with two digits
          img_num = n[i][5] + n[i][6]
      else: #number with one digit
        img_num = n[i][5]
        
      id_img = [int(s) for s in img_num.split() if s.isdigit()]
      id_img = id_img[0]    # convert single-element list of int in a single int
    
  
      x = X_labels[id_img,:].astype('int')    #####Controllare che vada bene l'astype
      y = Y_labels[id_img,:].astype('int')
      train_mask = create_mask(x,y,EPSILON)
      train_mask = cv2.resize(train_mask, (DEFAULT_HEIGHT//STRIDE, DEFAULT_WIDTH//STRIDE))
      #train_mask = train_mask.reshape(512, 512, 1) # Add extra dimension for parity with train_img size [512 * 512 * 3]

      mask[i-c] = train_mask

    c+=batch_size
    if(c+batch_size>=len(os.listdir(img_folder))):
      c=0
      random.shuffle(n)
                  # print "randomizing again"
    yield img, mask
