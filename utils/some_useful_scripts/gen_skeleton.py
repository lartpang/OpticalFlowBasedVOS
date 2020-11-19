# -*- coding: utf-8 -*-
# @Time    : 2020/9/26
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : gen_skeleton.py
# @Project : OpticalFlowBasedVOS
# @GitHub  : https://github.com/lartpang
import os

import cv2
import mmcv
import numpy as np
from skimage import morphology

davis_root = '/home/lart/Datasets/VideoSeg/DAVIS-2017-trainval-480p/DAVIS' \
             '/Annotations/480p'
save_root = '/home/lart/Datasets/VideoSeg/DAVIS-2017-trainval-480p/DAVIS' \
            '/SKAnnotations/480p'

for video_name in sorted(os.listdir(davis_root)):
    print(video_name, '...')
    video_path = os.path.join(davis_root, video_name)
    frame_name_list = sorted(os.listdir(video_path))
    for frame_name in frame_name_list:
        image_path = os.path.join(video_path, frame_name)
        save_path = os.path.join(save_root, video_name, frame_name)

        img = mmcv.imread(image_path, flag='grayscale')
        img[img > 0] = 1
        img = img.astype(np.uint8)
        mideanblured_img = cv2.medianBlur(img, 5)
        dilated_img = cv2.dilate(mideanblured_img,
                                 kernel=np.ones((7, 7), np.uint8),
                                 iterations=1)
        sk = morphology.skeletonize(dilated_img).astype(np.uint8)
        dialted_sk = cv2.dilate(sk, kernel=np.ones((3, 3), np.uint8),
                                iterations=1)
        mmcv.imwrite(dialted_sk.astype(np.uint8) * 255, save_path,
                     auto_mkdir=True)
