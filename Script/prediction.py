import numpy as np

def argmax_predict(predictions, stride):
    
    num_joints = predictions.shape[2]
    pose = []
    for joint_idx in range(num_joints):
        maxloc = np.unravel_index(np.argmax(predictions[:, :, joint_idx]), predictions[:, :, joint_idx].shape)
        pos_f8 = (np.array(maxloc).astype('float') * stride + 0.5 * stride)
        pose.append(np.hstack((pos_f8[::-1], [predictions[maxloc][joint_idx]])))
    return np.array(pose)
    