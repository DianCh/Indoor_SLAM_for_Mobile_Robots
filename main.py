import SLAM
import load_data as ld


def main():
    folder = "data/"
    number = 0
    train_joint = "train_joint" + str(number)
    train_lidar = "train_lidar" + str(number)
    test_joint = "test_joint"
    test_lidar = "test_lidar"

    title = "train data " + str(number)
    # title = "test data"

    joint = ld.get_joint(folder + train_joint)
    lidar = ld.get_lidar(folder + train_lidar)
    # joint = ld.get_joint(test_joint)
    # lidar = ld.get_lidar(test_lidar)

    map = SLAM.complete_SLAM(lidar, joint, title)

    return map


if __name__ == "__main__":
    map = main()
