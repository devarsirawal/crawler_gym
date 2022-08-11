import numpy as np
def inverse_rotation(p2,alpha,beta,gamma):  
    """
    Inverse rotation from a point p2 in global 3D reference frame 
    to a point p1 in the local (robot) reference frame.
 
    Input
    :param p2: A 3 element array containing the position of a point in the 
               global reference frame (xG,yG,zG)
    :param alpha: The roll angle (radians) - Rotation around the x-axis
    :param beta: The pitch angle (radians) - Rotation around the y-axis
    :param alpha: The yaw angle (radians) - Rotation around the z-axis
 
    Output
    :return: A 3 element array containing the position of a point in the 
             local reference frame (xL,yL,zL) 
 
    """
    # First row of the inverse rotation matrix
    r00 = np.cos(gamma) * np.cos(beta)
    r01 = np.sin(gamma) * np.cos(beta)
    r02 = -np.sin(beta)
     
    # Second row of the inverse rotation matrix 
    r10 = np.cos(gamma) * np.sin(beta) * np.sin(alpha) - np.sin(gamma) * np.cos(alpha)
    r11 = np.sin(gamma) * np.sin(beta) * np.sin(alpha) + np.cos(gamma) * np.cos(alpha)  
    r12 = np.cos(beta) * np.sin(alpha)  
     
    # Third row of the inverse rotation matrix  
    r20 = np.cos(gamma) * np.sin(beta) * np.cos(alpha) + np.sin(gamma) * np.sin(alpha)
    r21 = np.sin(gamma) * np.sin(beta) * np.cos(alpha) - np.cos(gamma) * np.sin(alpha)
    r22 = np.cos(beta) * np.cos(alpha)
     
    # 3x3 inverse rotation matrix
    inv_rot_matrix = np.array([[r00, r01, r02],
                               [r10, r11, r12],
                               [r20, r21, r22]]) 
                            
    return inv_rot_matrix @ p2
