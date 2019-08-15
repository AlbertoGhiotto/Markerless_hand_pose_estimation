import cv2
import numpy as np

STRIDE = 8
EPSILON = 8     ## Questo valore poi andrebbe un po' aggiustato
NUM_JOINTS = 16


def read_csv_labels():
  file = 'https://raw.githubusercontent.com/AlbertoGhiotto/group_project/master/Dataset/dataset.csv'
  labels = np.loadtxt(file, delimiter=',')
  X_labels = labels[:, 0:15]
  Y_labels = labels[:, 16:31]
  return X_labels, Y_labels


def create_mask(x,y,epsilon):                   ############# Questo poi forse sarebbe meglio metterlo come circonferenza intorno al punto invece di un quadrato
  mask = np.zeros((512/STRIDE, 512/STRIDE, NUM_JOINTS)).astype('float')
  for joint in range(NUM_JOINTS):
    for i in range(-epsilon/STRIDE, epsilon/STRIDE):
      for j in range(-epsilon/STRIDE, epsilon/STRIDE):
        mask[x[joint]/STRIDE + i, y[joint]/STRIDE + j] = 1
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
     
      ################ OLD ONE ##################
      #img_num = n[i][4] + n[i][5] + n[i][6]   #takes the last three elements of the string i.e the number
      #id_img = [int(s) for s in img_num.split() if s.isdigit()]
      #id_img = id_img[0]    # convert single-element list of int in a single int
      ###########################################     
  
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
