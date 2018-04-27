import numpy as np


'''
Rotate by angle theta, in 2D
'''
def Rot_2D(theta):
    return np.array([[np.cos(theta), - np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


'''
Rotate around Z by angle yaw, in 3D
'''
def Rot_Z(yaw):
    return np.array([[np.cos(yaw), - np.sin(yaw), 0],
                     [np.sin(yaw), np.cos(yaw),   0],
                     [0,           0,             1]])


'''
Rotate around Y by angle pitch, in 3D
'''
def Rot_Y(pitch):
    return np.array([[np.cos(pitch),   0, np.sin(pitch)],
                     [0,               1, 0],
                     [- np.sin(pitch), 0, np.cos(pitch)]])


'''
Rotate around X by angle roll, in 3D
'''
def Rot_X(roll):
    return np.array([[1, 0,            0],
                     [0, np.cos(roll), - np.sin(roll)],
                     [0, np.sin(roll), np.cos(roll)]])


'''
Combine R and t as T, for 2D motion
'''
def SE_2(Rot, t):
    T = np.concatenate((Rot, t), axis=1)
    T = np.concatenate((T, np.array([[0, 0, 1]])), axis=0)
    return T


'''
Combine R and t as T, for 3D motion
'''
def SE_3(Rot_3D, t):
    T = np.concatenate((Rot_3D, t), axis=1)
    T = np.concatenate((T, np.array([[0, 0, 0, 1]])), axis=0)
    return T


'''
Get T representing current pose, i.e., x, y, theta
'''
def SE_2_from_pose(pose):
    Rot = Rot_2D(pose[2])
    t = pose[:-1]
    T = SE_2(Rot, t)
    return T


'''
Apply delta x, delta y, delta theta on xy and theta
'''
def smart_plus(pose, delta):
    Rot = Rot_2D(pose[2])
    xy = pose[:2]
    # Create a new array to leave the passed in pose intact
    pose_next = np.zeros(3,)
    pose_next[:2] = xy + np.dot(Rot, delta[:2])
    pose_next[2] = pose[2] + delta[2]

    return pose_next


'''
Derive delta x, delta y, delta theta from two successive poses
'''
def smart_minus(pose_curr, pose_next):
    delta = np.zeros(3)
    Rot = Rot_2D(pose_curr[2])
    delta[:2] = np.dot(Rot.T, pose_next[:2] - pose_curr[:2])
    delta[2] = pose_next[2] - pose_curr[2]

    return delta

