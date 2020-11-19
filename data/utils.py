# -*- coding: utf-8 -*-
# @Time    : 2020/8/19
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : utils.py.py
# @Project : OpticalFlowBasedVOS
# @GitHub  : https://github.com/lartpang
import json
import os

import cv2
import mmcv
import numpy as np
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super(DataLoaderX, self).__iter__())


def read_data_dict_from_dir(dir_path: dict, data_set: str) -> dict:
    img_dir = dir_path["jpeg"]['path']
    img_suffix = dir_path["jpeg"]['suffix']
    mask_dir = dir_path["anno"]['path']
    mask_suffix = dir_path["anno"]['suffix']
    flow_dir = dir_path["flow"]['path']
    flow_suffix = dir_path["flow"]['suffix']
    set_path = os.path.join(dir_path["sets"]['path'], f"{data_set}{dir_path['sets']['suffix']}")

    video_name_list = []
    with open(set_path, encoding="utf-8", mode="r") as f:
        line = f.readline()
        while line:
            video_name_list.append(line.strip())
            line = f.readline()

    total_image_path_list = []
    total_mask_path_list = []
    total_flow_path_list = []
    for video_name in sorted(video_name_list):
        video_jpeg_path = os.path.join(img_dir, video_name)
        frame_name_list = sorted(os.listdir(video_jpeg_path))
        for idx, frame_name in enumerate(frame_name_list):
            frame_name = frame_name[:-4]
            total_image_path_list.append(dict(path=os.path.join(video_jpeg_path, frame_name + img_suffix),
                                              idx=idx))
            total_mask_path_list.append(dict(path=os.path.join(mask_dir, video_name, frame_name + mask_suffix),
                                             idx=idx))
            total_flow_path_list.append(dict(path=os.path.join(flow_dir, video_name, frame_name + flow_suffix),
                                             idx=idx))
    return dict(
        root=dir_path['root'],
        jpeg=total_image_path_list,
        anno=total_mask_path_list,
        flow=total_flow_path_list,
        name_list=video_name_list,
    )


def read_data_list_form_txt(path: str) -> list:
    line_list = []
    with open(path, encoding="utf-8", mode="r") as f:
        line = f.readline()
        while line:
            line_list.append(line.strip())
            line = f.readline()
    return line_list


def read_data_dict_from_json(json_path: str) -> dict:
    with open(json_path, mode="r", encoding="utf-8") as openedfile:
        data_info = json.load(openedfile)
    return data_info


def _flow_to_direction_and_magnitude(flow, unknown_thr=1e6):
    """Convert flow map to RGB image.

    Args:
        flow (ndarray): Array of optical flow.
        unknown_thr (str): Values above this threshold will be marked as
            unknown and thus ignored.

    Returns:
        ndarray: RGB image that can be visualized.
    """
    assert flow.ndim == 3 and flow.shape[-1] == 2
    color_wheel = mmcv.make_color_wheel()
    assert color_wheel.ndim == 2 and color_wheel.shape[1] == 3
    num_bins = color_wheel.shape[0]

    dx = flow[:, :, 0].copy()
    dy = flow[:, :, 1].copy()

    ignore_inds = (np.isnan(dx) | np.isnan(dy) |
                   (np.abs(dx) > unknown_thr) | (np.abs(dy) > unknown_thr))
    dx[ignore_inds] = 0
    dy[ignore_inds] = 0

    flow_magnitude = np.sqrt(dx ** 2 + dy ** 2)
    if np.any(flow_magnitude > np.finfo(float).eps):
        max_rad = np.max(flow_magnitude)
        dx /= max_rad
        dy /= max_rad

    flow_magnitude = np.sqrt(dx ** 2 + dy ** 2)
    flow_direction = np.arctan2(-dy, -dx) / np.pi  # -1,1

    bin_real = (flow_direction + 1) / 2 * (num_bins - 1)  # [0,num_bins-1)
    bin_left = np.floor(bin_real).astype(int)
    bin_right = (bin_left + 1) % num_bins
    w = (bin_real - bin_left.astype(np.float32))[..., None]
    flow_img = (1 - w) * color_wheel[bin_left, :] + \
               w * color_wheel[bin_right, :]
    direction_map = flow_img.copy()
    small_ind = flow_magnitude <= 1
    flow_img[small_ind] = 1 - flow_magnitude[small_ind, None] * \
                          (1 - flow_img[small_ind])
    flow_img[np.logical_not(small_ind)] *= 0.75
    flow_img[ignore_inds, :] = 0

    return dict(flow=flow_img,
                direction=direction_map,
                magnitude=flow_magnitude)


def read_flow_array(path: str, return_info, to_normalize=False):
    """
    :param path: 支持flo(mmcv)和png(opencv)数据
    :param return_info: 对于flo数据，支持三种选择(flow,direction,magnitude)，对于png数据直接读取，返回键为flow
    :param to_normalize: 仅对flo数据的magnitude数据有效
    :return: 0~1
    """
    if path.endswith('.flo'):
        flow_array = mmcv.flowread(path)
        split_flow = _flow_to_direction_and_magnitude(flow_array)
        if not isinstance(return_info, (tuple, list)):
            return_info = [return_info]

        return_array = dict()
        for k in return_info:
            data_array = split_flow[k]
            if k == 'magnitude' and to_normalize:
                data_array = (data_array - data_array.min()) / \
                             (data_array.max() - data_array.min())
            return_array[k] = data_array
    else:
        return_array = dict(flow=read_color_array(path))
    return return_array


def read_color_array(path: str):
    assert path.endswith('.jpg') or path.endswith('.png')
    bgr_array = cv2.imread(path, cv2.IMREAD_COLOR)
    rgb_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
    return rgb_array


def read_binary_array(path: str, thr=-1, to_normalize=False):
    assert path.endswith('.jpg') or path.endswith('.png')
    gray_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    gray_array = gray_array.astype(np.float32)

    if to_normalize:
        gray_array = (gray_array - gray_array.min()) / (
                gray_array.max() - gray_array.min())
    if thr >= 0:
        gray_array = np.where(gray_array > thr, 1, 0).astype(np.float32)
    return gray_array
