import torch
import torch.utils.data as data

import os
import os.path

import numpy as np

import math


object_list = {'Car': 0, 'Van': 1, 'Truck': 2}
class_list = ['Car', 'Van', 'Truck']

bc = {}
bc['minX'] = 0;
bc['maxX'] = 80;
bc['minY'] = -40;
bc['maxY'] = 40
bc['minZ'] = -2;
bc['maxZ'] = 1.25


def interpret_kitti_label(bbox):
    w, h, l, y, z, x, yaw = bbox[8:15]
    y = -y
    yaw = (yaw + np.pi / 2)

    return x, y, w, l, yaw


def get_target2(label_file):
    target = np.zeros([50, 5], dtype=np.float32)

    with open(label_file, 'r') as f:
        lines = f.readlines()

    num_obj = len(lines)
    index = 0

    for j in range(num_obj):
        obj = lines[j].strip().split(' ')
        obj_class = obj[0].strip()

        if obj_class in class_list:
            bbox = []
            bbox.append(object_list[obj_class])
            bbox.extend([float(e) for e in obj[1:]])

            x, y, w, l, yaw = interpret_kitti_label(bbox)

            location_x = x
            location_y = y

            if (location_x > 0) & (location_x < 40) & (location_y > -40) & (location_y < 40):
                x = (y + 40) / 80
                y = x / 40

                length = float(l) / 80
                width = float(w) / 40

                target[index][0] = x - length / 2   # we should put this in [0,1], so divide max_size  80 m
                target[index][1] = y - width / 2  # make sure target inside the covering area (0,1)

                target[index][2] = x + length / 2
                target[index][3] = y + width / 2  # get target width, length

                # target[index][5] = math.sin(float(yaw))  # complex YOLO   Im
                # target[index][6] = math.cos(float(yaw))  # complex YOLO   Re

                for i in range(len(class_list)):
                    if obj_class == class_list[i]:  # get target class
                        target[index][4] = i

                index = index + 1

    # print(label_file)
    # print(target)
    target = target[:index,:]

    return target


def removePoints(PointCloud, BoundaryCond):
    # Boundary condition
    minX = BoundaryCond['minX'];
    maxX = BoundaryCond['maxX']
    minY = BoundaryCond['minY'];
    maxY = BoundaryCond['maxY']
    minZ = BoundaryCond['minZ'];
    maxZ = BoundaryCond['maxZ']

    # Remove the point out of range x,y,z
    mask = np.where((PointCloud[:, 0] >= minX) & (PointCloud[:, 0] <= maxX) & (PointCloud[:, 1] >= minY) & (
                PointCloud[:, 1] <= maxY) & (PointCloud[:, 2] >= minZ) & (PointCloud[:, 2] <= maxZ))
    PointCloud = PointCloud[mask]

    PointCloud[:, 2] = PointCloud[:, 2] + 2
    return PointCloud


def makeBVFeature(PointCloud_, BoundaryCond, Discretization):
    # 1024 x 1024 x 3
    Height = 300 + 1
    Width = 300 + 1

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / Discretization))
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / Discretization) + Width / 2)

    # sort-3times
    indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[indices]

    # Height Map
    heightMap = np.zeros((Height, Width))

    _, indices = np.unique(PointCloud[:, 0:2], axis=0, return_index=True)
    PointCloud_frac = PointCloud[indices]
    # some important problem is image coordinate is (y,x), not (x,y)
    heightMap[np.int_(PointCloud_frac[:, 0]), np.int_(PointCloud_frac[:, 1])] = PointCloud_frac[:, 2]

    # Intensity Map & DensityMap
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))

    _, indices, counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_top = PointCloud[indices]

    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))

    intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts
    """
    plt.imshow(densityMap[:,:])
    plt.pause(2)
    plt.close()
    plt.show()
    plt.pause(2)
    plt.close()
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    plt.imshow(intensityMap[:,:])
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    """
    RGB_Map = np.zeros((Height, Width, 3))
    RGB_Map[:, :, 0] = densityMap  # r_map
    RGB_Map[:, :, 1] = heightMap  # g_map
    RGB_Map[:, :, 2] = intensityMap  # b_map

    save = np.zeros((512, 1024, 3))
    save = RGB_Map[0:512, 0:1024, :]
    # misc.imsave('test_bv.png',save[::-1,::-1,:])
    # misc.imsave('test_bv.png',save)
    return save


class KittiDataset(data.Dataset):

    def __init__(self, root, set='train', type='velodyne_train'):
        self.type = type
        self.root = root
        self.data_path = os.path.join(root)
        self.lidar_path = os.path.join(self.data_path, "velodyne")
        self.image_path = os.path.join(self.data_path, "image_2")
        self.calib_path = os.path.join(self.data_path, "calib")
        self.label_path = os.path.join(self.data_path, "label_2")

        self.name = "KITTI"

        self.file_list = []

        with open(os.path.join(self.data_path, '%s.txt' % set)) as f:
            file_list_cand = f.read().splitlines()

            for i in file_list_cand:
                label_file = self.label_path + '/' + i + '.txt'
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    num_obj = len(lines)

                    for j in range(num_obj):
                        obj = lines[j].strip().split(' ')
                        obj_class = obj[0].strip()

                        if obj_class in class_list:
                            bbox = []
                            bbox.append(object_list[obj_class])
                            bbox.extend([float(e) for e in obj[1:]])

                            x, y, w, l, yaw = interpret_kitti_label(bbox)

                            location_x = x
                            location_y = y

                            if (location_x > 0) & (location_x < 40) & (location_y > -40) & (location_y < 40):
                                self.file_list.append(i)
                                break

    def __getitem__(self, i):

        lidar_file = self.lidar_path + '/' + self.file_list[i] + '.bin'
        label_file = self.label_path + '/' + self.file_list[i] + '.txt'

        target = get_target2(label_file)

        a = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

        b = removePoints(a, bc)

        data = makeBVFeature(b, bc, 40 / 150)  # (512, 1024, 3)

        return data, target


    def __len__(self):
        return len(self.file_list)
