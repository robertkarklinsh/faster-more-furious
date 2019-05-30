from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from data.kitti import *
from data.config import kitti
import torch.utils.data as data
from ssd import build_ssd

import matplotlib.image as mpimg
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd300_COCO_10000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, thresh):
    for i in range(1):
        filename = os.path.join(save_folder, "img_{%d}.png" % i)

        data, target = testset.__getitem__(i)

        mpimg.imsave(os.path.join(save_folder, 'sample.png'), data / data.max())

        data = torch.tensor(data)
        data = data.reshape(1, 3, 301, 301).float()

        if cuda:
            data = data.cuda()

        y = net(data)      # forward pass
        y = y.squeeze()

        print(y.shape)
        print(y[1][0])

        test = y[1][0].cpu().detach().numpy()

        image = cv2.imread(os.path.join(save_folder, 'sample.png'))

        x = int(test[0] * 301.0)  # 32 cell = 1024 pixels
        y = int(test[1] * 301.0)  # 16 cell = 512 pixels
        width = int(test[2] * 301.0)  # 32 cell = 1024 pixels
        height = int(test[3] * 301.0)  # 16 cell = 512 pixels

        local_coords = np.array([
            [-width / 2, -height / 2],
            [-width / 2, height / 2],
            [width / 2, height / 2],
            [width / 2, -height / 2]
        ], np.int32)

        # local_rotation = np.array([
        #     [np.cos(angle), -np.sin(angle)],
        #     [np.sin(angle), np.cos(angle)]
        # ])

        transition = np.array([
            [x, y]
        ])

        coords = local_coords + transition
        coords = coords.astype(np.int32)
        coords = coords.reshape((-1, 1, 2))

        cv2.polylines(image, [coords], True, (255, 0, 0), 1)

        # cv2.polylines(img, [pred_coords], True, (255, 0, 0), 2)

        for i in target:
            x = int(i[0] * 301.0)  # 32 cell = 1024 pixels
            y = int(i[1] * 301.0)  # 16 cell = 512 pixels
            width = int(i[2] * 301.0)  # 32 cell = 1024 pixels
            height = int(i[3] * 301.0)  # 16 cell = 512 pixels

            local_coords = np.array([
                [-height / 2, -width / 2],
                [height / 2, -width / 2],
                [height / 2, width / 2],
                [-height / 2, width / 2]
            ], np.int32)

            # local_rotation = np.array([
            #     [np.cos(angle), -np.sin(angle)],
            #     [np.sin(angle), np.cos(angle)]
            # ])

            transition = np.array([
                [x, y]
            ])

            coords = local_coords + transition
            coords = coords.astype(np.int32)
            coords = coords.reshape((-1, 1, 2))

            cv2.polylines(image, [coords], True, (0, 0, 255), 1)

        mpimg.imsave(filename, image)


        # detections = y.data
        # scale each detection back up to the image
        # scale = torch.Tensor([img.shape[1], img.shape[0],
        #                      img.shape[1], img.shape[0]])
        # pred_num = 0
        # for i in range(detections.size(1)):
        #     j = 0
        #     while detections[0, i, j, 0] >= 0.6:
        #         if pred_num == 0:
        #             with open(filename, mode='a') as f:
        #                 f.write('PREDICTIONS: '+'\n')
        #         score = detections[0, i, j, 0]
        #         label_name = labelmap[i-1]
        #         pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
        #         coords = (pt[0], pt[1], pt[2], pt[3])
        #         pred_num += 1
        #         with open(filename, mode='a') as f:
        #             f.write(str(pred_num)+' label: '+label_name+' score: ' +
        #                     str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
        #         j += 1


def test_voc():
    # load net
    cfg = kitti
    net = build_ssd('test', 300, cfg['num_classes']) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data

    testset = KittiDataset(root='/data/KITTI_OBJECTS_3D/training', set='train')
    if args.cuda:
        net = net.cuda()
        # cudnn.benchmark = True

    # evaluation
    test_net(args.save_folder, net, args.cuda, testset, thresh=args.visual_threshold)

if __name__ == '__main__':
    test_voc()
