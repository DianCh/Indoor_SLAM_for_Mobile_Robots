# Indoor SLAM for Mobile Robots
This repository contains implementation of a LIDAR-based 2-D FAST SLAM with particle filtering. All sensor data came from a real humanoid robot wandering in the Engineering buildings at Penn. 2-D occupancy grid maps are developed in the process of SLAM, which match the actual floorplans.

The dataset is comprised of **LIDAR scans**, **odometry**, **physical configurations** of the robot (neck angles are time-varying) and additional **RGBD image** for optional point cloud visualization.

----

## Run tests
To run the SLAM, simply execute the main script:

- main.py

The entire SLAM may take a long time, up to **20 to 30 minutes** on the provided dataset.

Feel free to play around with the hyper-parameters anywhere in the scripts.

## Dependencies
For you to run the tests, python 3 with normal scientific packages (numpy, matplotlib, etc.) would suffice.

## Sample Results
Here is the resulted map and trajectory for one of the dataset:

<img src="https://github.com/DianCh/Learning_in_Robotics/blob/master/4_Indoor_SLAM/results/train_0.png" width="600">

A detailed report for this project can also be found [here](https://github.com/DianCh/Indoor_SLAM_for_Mobile_Robots/blob/master/results/report.pdf).