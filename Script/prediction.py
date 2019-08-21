import numpy as np
import cv2

def argmax_predict(predictions, stride):
    
    num_joints = predictions.shape[3]
    pose = []
    for joint_idx in range(num_joints):
        maxloc = np.unravel_index(np.argmax(predictions[:, :, joint_idx]), predictions[:, :, joint_idx].shape)
        pos_f8 = (np.array(maxloc).astype('float') * stride + 0.5 * stride)
        pose.append(np.hstack((pos_f8[::-1], [predictions[maxloc][joint_idx]])))
    return np.array(pose)


def visualize(image, pose):
    
    cv2.imshow('pose', image)  
    for joint in range(pose.shape[0]):
        cx = pose[joint][0]
        cy = pose[joint][1]  
  
        cv2.circle(image,(int(cx),int(cy)),10,(255,255,255),-11)
