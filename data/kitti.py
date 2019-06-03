import torch
import torch.utils.data as data

import os
import os.path

import numpy as np

import math


object_list = {'Car':0, 'Van':1, 'Truck':2}
class_list = ['Car', 'Van' , 'Truck']

bc={}
bc['minX'] = 0; bc['maxX'] = 80; bc['minY'] = -40; bc['maxY'] = 40
bc['minZ'] =-2; bc['maxZ'] = 1.25

def interpret_kitti_label(bbox):
    w, h, l, y, z, x, yaw = bbox[8:15]
    y = -y
    yaw =  (yaw + np.pi / 2)

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

            if (x > bc['minX']) & (x < bc['maxX']) & (y > bc['minY']) & (y < bc['maxY']):
                pos_x = np.int_(np.floor(y / (80 / 300)) + 301 / 2)
                pos_y = np.int_(np.floor(x / (80 / 300)))

                width = np.int_(np.floor(w / (55 / 301)))
                height = np.int_(np.floor(l / (55 / 301)))

                # xmin, ymin, xmax, ymax,

                target[index][0] = (pos_x - width / 2) / 301
                target[index][1] = (pos_y - height / 2) / 301
                target[index][2] = (pos_x + width / 2) / 301
                target[index][3] = (pos_y + height / 2) / 301

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


def makeBVFeature(PointCloud_, Discretization):
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

    # save = np.zeros((512, 1024, 3))
    # save = RGB_Map[0:512, 0:1024, :]
    # misc.imsave('test_bv.png',save[::-1,::-1,:])
    # misc.imsave('test_bv.png',save)
    return RGB_Map

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

        data = makeBVFeature(b, 80 / 300)  # (512, 1024, 3)

        return data, target


    def __len__(self):
        return len(self.file_list)

import matplotlib.image as mpimg
import cv2

def test_voxelization():
    dataset = KittiDataset(root='/data/KITTI_OBJECTS_3D/training', set='train')

    data, target = dataset.__getitem__(3)

    save_folder = '../eval/'
    mpimg.imsave(os.path.join(save_folder, 'sample.png'), data / data.max())

    image = cv2.imread(os.path.join(save_folder, 'sample.png'))

    for i in target:
        i = (i * 301).astype(np.int)

        cv2.rectangle(image, (i[0],i[1]), (i[2],i[3]), (0, 0, 255))

    mpimg.imsave(os.path.join(save_folder, 'sample.png'), image)

if __name__ == '__main__':
    test_voxelization()