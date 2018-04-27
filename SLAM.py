import numpy as np
import Transformations as Tf
import cv2
import matplotlib.pyplot as plt


def complete_SLAM(lidar, joint, title):
    # Initialize map
    map = init_map(size=22, resolution=0.1)

    # Initialize particles
    N = 500
    particles = np.zeros((4, N))
    particles[3, :] = 1.0 / N
    best_particle = np.zeros((4, 0))

    # Preprocess lidar yaw, compute the bias
    lidar_yaw_bias, l = 0, 50
    for i in range(l):
        lidar_yaw_bias += lidar[i]["rpy"][0, 2]
    lidar_yaw_bias /= l

    # plot animation map
    plt.figure()
    jump = 10
    # SLAM sequentially
    for i in range(0, len(lidar)-jump, jump):
        print(i)
        # Time stamp of lidar data
        t_lidar = lidar[i]["t"]

        # Find the matching time stamp in joint data
        t_joint_ind = match_time(t_lidar, joint["ts"])

        # Bearings of rays in head frame
        bearings = np.array([np.arange(-135, 135.25, 0.25) * np.pi / 180.0])

        # First get rid of possible invalid rays, i.e., too close or too far
        zt = lidar[i]["scan"]
        valid_ind = np.logical_and(zt < 30, zt > 0.1)
        zt = zt[valid_ind][np.newaxis, :]
        bearings = bearings[valid_ind][np.newaxis, :]

        head_yaw = joint["head_angles"][0, t_joint_ind]
        head_pitch = joint["head_angles"][1, t_joint_ind]
        odometry_curr = lidar[i]["pose"][0, :]          # Reduce to 1-D array
        odometry_next = lidar[i + jump]["pose"][0, :]

        # Substitute yaw in odometry with yaw in lidar.rpy
        odometry_curr[2] = lidar[i]["rpy"][0, 2] - lidar_yaw_bias
        odometry_next[2] = lidar[i + jump]["rpy"][0, 2] - lidar_yaw_bias

        # Use the lidar.rpy information
        rpy = lidar[i]["rpy"]

        # Perform one step of SLAM
        map, particles, MLE_particle = SLAM_one_step(particles, zt, rpy, bearings, map, head_yaw, head_pitch, odometry_curr, odometry_next)
        best_particle = np.concatenate((best_particle, MLE_particle[:, np.newaxis]), axis=1)

        # plot the map
        plt.imshow(map["log_map"], cmap="hot")
        i_ind, j_ind = get_end_cells(particles[:2, :], map["res"], map["xmin"], map["ymin"], exclude_ground=False)
        plt.scatter(j_ind, i_ind)
        plt.pause(0.0001)
        plt.clf()

    traj_i, traj_j = get_end_cells(best_particle[:2, :], map["res"], map["xmin"], map["ymin"], exclude_ground=False)
    plt.close()
    plt.figure()
    plt.imshow(map["log_map"], cmap="hot")
    plt.plot(traj_j, traj_i)
    plt.title(title)
    plt.xlabel("x (dm)")
    plt.ylabel("y (dm)")
    plt.show()

    return map


def SLAM_one_step(particles, zt, rpy, bearings, map, head_yaw, head_pitch, odometry_curr, odometry_next):
    # Find particle with the highest weight
    MLE_ind = np.argmax(particles[3, :])
    MLE_particle = particles[:, MLE_ind]

    # Perform mapping using this best particle
    map = mapping(zt=zt, bearings=bearings, particle=MLE_particle, rpy=rpy, head_yaw=head_yaw, head_pitch=head_pitch, map=map)

    # Predict new positions of particles
    particles, _ = localization_prediction(particles, odometry_curr, odometry_next)
    # _, particles = localization_prediction(particles, odometry_curr, odometry_next)

    # Update weights of particles
    particles = localization_update(particles, zt, rpy, bearings, map, head_yaw, head_pitch)

    return map, particles, MLE_particle


def mapping(zt, bearings, particle, rpy, head_yaw, head_pitch, map):
    # Transform rays into global frame
    # XY, Z = transform_scan(zt, particle, head_yaw, head_pitch, bearings)
    XY, Z = transform_scan(zt, particle, rpy, head_yaw, head_pitch, bearings)

    # Detect which rays hit the ground
    thresh = 0.1
    ground = Z < thresh
    above_ground = np.logical_not(ground)

    # Get cells for obstacles only
    i_ind_obt, j_ind_obt = get_end_cells(XY,
                                         exclude_ground=True,
                                         above_ground=above_ground,
                                         resolution=map["res"],
                                         x_min=map["xmin"],
                                         y_min=map["ymin"])
    # Get cells for endpoints of all rays
    i_ind_all, j_ind_all = get_end_cells(XY,
                                         exclude_ground=False,
                                         above_ground=None,
                                         resolution=map["res"],
                                         x_min=map["xmin"],
                                         y_min=map["ymin"])
    print(i_ind_obt.shape[0], i_ind_all.shape[0])

    # Get cell for the particle
    x_ind = int((particle[0] + np.abs(map["xmin"])) / map["res"])
    y_ind = int((np.abs(map["ymin"]) - particle[1]) / map["res"])

    # Fill the log-odds of free cells
    poly_vertices = np.array([j_ind_all, i_ind_all]).T
    poly_vertices = np.concatenate((poly_vertices, np.array([[x_ind, y_ind]])), axis=0)

    mask = np.zeros(map["log_map"].shape)
    cv2.fillPoly(mask, [poly_vertices], color=1)

    log_odds_obt, log_odds_free = 15, 5
    map["log_map"] = map["log_map"] - mask * log_odds_free

    # Add log-odds to the obstacles
    map["log_map"][i_ind_obt, j_ind_obt] = map["log_map"][i_ind_obt, j_ind_obt] + log_odds_free + log_odds_obt

    # Cap the log-odds at upper and lower bounds
    map["log_map"] = np.clip(map["log_map"], a_min=-255, a_max=255)

    return map


def localization_prediction(particles, odometry_curr, odometry_next):
    # The number of particles
    N = particles.shape[1]
    particles_dead = np.copy(particles)

    # Compute the motion from odometry
    delta = Tf.smart_minus(odometry_curr, odometry_next)
    noise = np.zeros((N, 3))

    if np.linalg.norm(delta) > 0.01:
        # Generate random motion noise
        scale = np.array([2, 2, 5 * np.pi / 180]) * np.linalg.norm(delta)
        # scale = np.array([1, 1, 10 * np.pi / 180]) * np.linalg.norm(delta)
        cov = np.diag(scale)
        mean = np.zeros(3)
        noise = np.random.multivariate_normal(mean, cov, N)

    # Transform each particle
    for i in range(N):
        # Apply motion, i.e., only dead reckoning
        particles_dead[:3, i] = Tf.smart_plus(particles[:3, i], delta)
        # Apply noise
        particles[:3, i] = Tf.smart_plus(particles_dead[:3, i], noise[i, :])

    return particles, particles_dead


def localization_update(particles, zt, rpy, bearings, map, head_yaw, head_pitch):
    # The number of particles
    N = particles.shape[1]
    corrs = np.zeros(N,)

    # Examine each particle
    for i in range(N):
        # Transform rays into global frame
        XY, Z = transform_scan(zt, particles[:3, i], rpy, head_yaw, head_pitch, bearings)

        corrs[i] = correlation(XY, Z, map, log_odds_thresh=100)

    # Update the weights of particles
    particles = update_weights(particles, corrs)

    # Compute the number of effective particles
    N_effective = 1 / np.sum(np.square(particles[3, :]))

    # Resample if necessary
    N_thresh = 8
    print(N_effective)
    if N_effective < N_thresh:
        particles = resample(particles)

    return particles


def init_map(size=20, resolution=0.05):
    MAP = {}
    MAP['res'] = resolution  # meters
    MAP['xmin'] = -size  # meters
    MAP['ymin'] = -size
    MAP['xmax'] = size
    MAP['ymax'] = size
    MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))  # cells
    MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))

    MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.int8)  # DATA TYPE: char or int8
    MAP['log_map'] = np.zeros((MAP['sizex'], MAP['sizey']))

    return MAP


# def transform_scan(zt, particle, rpy, head_yaw, head_pitch, lidar_bearings):
#     # Given configurations
#     t_center = np.array([[0], [0], [0.93]])
#     t_head = np.array([[0], [0], [0.33]])
#     t_lidar = np.array([[0], [0], [0.15]])
#
#     # Robot heading and xy position
#     t_xy = np.array([[particle[0]], [particle[1]], [0]])
#     robot_heading = particle[2]
#
#     # Transformation from world to body
#     T_wb = Tf.SE_3(Tf.Rot_Z(robot_heading), t_xy + t_center + t_head)
#
#     # Transformation from body to lidar
#     T1 = Tf.SE_3(Tf.Rot_Z(head_yaw) * Tf.Rot_Y(head_pitch), np.zeros((3, 1)))
#     T2 = Tf.SE_3(np.eye(3), t_lidar)
#     T_bl = T1 * T2
#
#     # Transform the hit points to world frame
#     M = zt.shape[1]
#     x = zt * np.cos(lidar_bearings)
#     y = zt * np.sin(lidar_bearings)
#     xyz_robo = np.concatenate((x, y, np.zeros((1, M)), np.ones((1, M))), axis=0)
#
#     XYZ = np.dot(T_wb, np.dot(T_bl, xyz_robo))[:3, :]
#     XY = XYZ[:2, :]     # 2-D array
#     Z = XYZ[2, :]       # 1-D array
#
#     return XY, Z

def transform_scan(zt, particle, rpy, head_yaw, head_pitch, lidar_bearings):
    r, p, y = rpy[0, 0], rpy[0, 1], particle[2]
    # r, p, y = 0, 0, particle[2]
    r11 = np.cos(y) * np.cos(p)
    r12 = np.cos(y) * np.sin(p) * np.sin(r) - np.sin(y) * np.cos(r)
    r13 = np.cos(y) * np.sin(p) * np.cos(r) + np.sin(y) * np.sin(r)

    r21 = np.sin(y) * np.cos(p)
    r22 = np.sin(y) * np.sin(p) * np.sin(r) + np.cos(y) * np.cos(r)
    r23 = np.sin(y) * np.sin(p) * np.cos(r) - np.cos(y) * np.sin(r)

    r31 = -np.sin(p)
    r32 = np.cos(p) * np.sin(r)
    r33 = np.cos(p) * np.cos(r)

    # transfer from world to body
    t_w2b = np.array([[r11, r12, r13, particle[0]],
                      [r21, r22, r23, particle[1]],
                      [r31, r32, r33, 0.93],
                      [0, 0, 0, 1]])

    # transfer from body to head
    t_b2h = np.array([[np.cos(head_yaw), -np.sin(head_yaw), 0, 0],
                      [np.sin(head_yaw), np.cos(head_yaw), 0, 0],
                      [0, 0, 1, 0.33],
                      [0, 0, 0, 1]])

    # transfer from head to lidar
    t_h2l = np.array([[np.cos(head_pitch), 0, np.sin(head_pitch), 0],
                      [0, 1, 0, 0],
                      [-np.sin(head_pitch), 0, np.cos(head_pitch), 0.15],
                      [0, 0, 0, 1]])

    T = np.einsum('ij,jk,kl->il', t_w2b, t_b2h, t_h2l)

    # Transform the hit points to world frame
    M = zt.shape[1]
    x = zt * np.cos(lidar_bearings)
    y = zt * np.sin(lidar_bearings)
    xyz_robo = np.concatenate((x, y, np.zeros((1, M)), np.ones((1, M))), axis=0)

    XYZ = np.dot(T, xyz_robo)[:3, :]
    XY = XYZ[:2, :]     # 2-D array
    Z = XYZ[2, :]       # 1-D array

    return XY, Z


def get_end_cells(XY, resolution, x_min, y_min, exclude_ground=False, above_ground=None):
    # If only need obstacles, filter out the hit points on the ground
    if exclude_ground:
        XY = XY[:, above_ground]

    x_ind = (XY[0, :] + np.abs(x_min)) / resolution
    x_ind = x_ind.astype(int)
    y_ind = (np.abs(y_min) - XY[1, :]) / resolution
    y_ind = y_ind.astype(int)

    i_ind = y_ind       # 1-D array
    j_ind = x_ind       # 1-D array

    return i_ind, j_ind


def correlation(XY, Z, map, log_odds_thresh=100):

    # Detect which rays hit the ground
    thresh = 0.1
    ground = Z < thresh
    above_ground = np.logical_not(ground)

    log_odds_map = np.copy(map["log_map"])

    # Determine the obstacle grids in the current map
    map_binary = log_odds_map > log_odds_thresh

    # Take only the log-odds of obstacle cells
    log_odds_map = log_odds_map * map_binary
    i_ind, j_ind = get_end_cells(XY, map["res"], map["xmin"], map["ymin"],
                                 exclude_ground=True, above_ground=above_ground)
    corr = np.sum(map_binary[i_ind, j_ind])

    return corr


def update_weights(particles, corrs):
    # Convert weights to log scale
    log_weights = np.log(particles[3, :]) + corrs * 20 / (np.max(corrs) + 1)

    log_weights_trans = log_weights - np.max(log_weights)

    # Update the weights
    log_weights = log_weights_trans - np.log(np.sum(np.exp(log_weights_trans)))
    weights = np.exp(log_weights)

    particles[3, :] = weights
    print(corrs)
    print(weights)

    return particles


def resample(particles):
    # The number of particles
    N = particles.shape[1]
    particles_new = np.copy(particles)

    # Using stratified sampling
    r = np.random.uniform(low=0, high=1.0 / N)
    c = particles[3, 0]
    i = 0
    for k in range(N):
        u = r + k * (1.0 / N)

        while u > c:
            i = i + 1
            c = c + particles[3, i]

        particles_new[:, k] = particles[:, i]

    # Reset the weights to uniform
    particles_new[3, :] = 1.0 / N

    return particles_new


def match_time(time_current, time_sequence):
    err = np.abs(time_sequence - time_current)
    index = np.argmin(err)

    return index

