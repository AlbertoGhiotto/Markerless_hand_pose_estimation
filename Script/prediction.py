import numpy as np
import cv2

def argmax_predict(predictions, stride):
    
    num_joints = predictions.shape[2]
    pose = []
    for joint_idx in range(num_joints):
        maxloc = np.unravel_index(np.argmax(predictions[:, :, joint_idx]), predictions[:, :, joint_idx].shape)
        pos_f8 = (np.array(maxloc).astype('float') * stride + 0.5 * stride)
        pose.append(np.hstack((pos_f8[::-1], [predictions[maxloc][joint_idx]])))
    return np.array(pose)


def visualize(image, pose):
   
  for joint in range(pose.shape[0]):
      cx = pose[joint][0]
      cy = pose[joint][1]  
      
      cv2.circle(image,(int(cx),int(cy)),5,(255,255,255),-1) #Draw a cirle onto the image
      cv2.putText(image, str(joint), (int(cx),int(cy)),cv2.FONT_ITALIC ,1, (255,0,0), 3,cv2.LINE_AA) #Puts joints number inside the cirle
      
  cv2_imshow(image)
