# -*- coding: utf-8 -*-
# @Time    : 2020/11/15
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : davis16.py
# @Project : OpticalFlowBasedVOS
# @GitHub  : https://github.com/lartpang

import os

_davis_root = "/home/lart/Datasets/VideoSeg/DAVIS-2017-trainval-480p/DAVIS"

DAVIS16_paths = dict(
    root=_davis_root,
    sets=dict(path=os.path.join(_davis_root, "ImageSets", "2016"), suffix='.txt'),
    jpeg=dict(path=os.path.join(_davis_root, "JPEGImages", "480p"), suffix='.jpg'),
    anno=dict(path=os.path.join(_davis_root, "Annotations", "480p"), suffix='.png'),
    flow=dict(path=os.path.join(_davis_root, "OriFlowImagesFromPWC", "480p"), suffix='.flo'),
    # flow=dict(path=os.path.join(_davis_root, "FlowImagesFromPWC", "480p"), suffix='.png'),
)
