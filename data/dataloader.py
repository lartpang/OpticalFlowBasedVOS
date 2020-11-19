import random
from collections import defaultdict

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch.nn.functional import interpolate
from torch.utils.data import Dataset

from config import arg_config
from data.utils import (
    DataLoaderX, read_binary_array, read_color_array,
    read_data_dict_from_dir, read_flow_array,
)
from utils.misc import construct_print, set_seed_for_lib


def get_hw(in_size):
    if new_size := in_size.get('hw', False):
        assert isinstance(new_size, int)
        full_new_size = (new_size, new_size)
    elif in_size.get('h', False) and in_size.get('w', False):
        assert isinstance(in_size['h'], int)
        assert isinstance(in_size['w'], int)
        full_new_size = (in_size['h'], in_size['w'])
    else:
        raise Exception(f"{in_size} error!")
    return full_new_size


class TestDataset(Dataset):
    def __init__(self, root: dict, in_size: dict):
        """
        :param root: 这里的root是实际对应的数据字典
        :param in_size:
        """
        new_data_dict = read_data_dict_from_dir(root, data_set="val")
        self.total_image_paths = new_data_dict["jpeg"]
        self.total_flow_paths = new_data_dict["flow"]
        self.total_mask_paths = new_data_dict["anno"]
        self.video_name_list = new_data_dict["name_list"]
        construct_print(f"Loading data from: {new_data_dict['root']}")
        assert len(self.total_image_paths) == len(self.total_mask_paths) == len(self.total_flow_paths)

        h, w = get_hw(in_size)
        self.transform = A.Compose(
            [
                A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            additional_targets=dict(flow='image')
        )

    def __getitem__(self, index):
        curr_img_path = self.total_image_paths[index]["path"]
        curr_flow_path = self.total_flow_paths[index]["path"]
        curr_mask_path = self.total_mask_paths[index]["path"]

        curr_img = read_color_array(curr_img_path)
        curr_flow = read_flow_array(curr_flow_path, return_info=['flow'], to_normalize=False)['flow'] * 255

        transformed = self.transform(image=curr_img, flow=curr_flow)
        curr_img = transformed["image"]
        curr_flow = transformed['flow']

        return dict(
            image=curr_img,
            flow=curr_flow,
            mask_path=curr_mask_path,
        )

    def __len__(self):
        return len(self.total_image_paths)


class TrainDataset(Dataset):
    def __init__(self, root: dict, in_size: dict):
        super(TrainDataset, self).__init__()
        self.scale_list = [1.0]
        self.scale_list.extend(in_size['extra_scales'])

        self.total_image_paths = []
        self.total_flow_paths = []
        self.total_mask_paths = []
        self.video_name_list = []
        for root_name, root_item in root:
            new_data_dict = read_data_dict_from_dir(root_item, data_set="train")
            construct_print(f"Loading data from {root_name}: {new_data_dict['root']}")
            self.total_image_paths += new_data_dict["jpeg"]
            self.total_flow_paths += new_data_dict['flow']
            self.total_mask_paths += new_data_dict["anno"]
            self.video_name_list += new_data_dict["name_list"]
        assert len(self.total_image_paths) == len(self.total_mask_paths) == len(self.total_flow_paths)

        h, w = get_hw(in_size)
        self.img_transform = A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5)
        self.joint_transform = A.Compose(
            [
                A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            additional_targets=dict(flow='image')
        )

    def __getitem__(self, index):
        curr_img_path = self.total_image_paths[index]["path"]
        curr_flow_path = self.total_flow_paths[index]["path"]
        curr_mask_path = self.total_mask_paths[index]["path"]

        curr_img = read_color_array(curr_img_path)
        curr_flow = read_flow_array(curr_flow_path, return_info=['flow'], to_normalize=False)['flow'] * 255
        curr_mask = read_binary_array(curr_mask_path, thr=0)

        curr_img = self.img_transform(image=curr_img)['image']
        transformed = self.joint_transform(image=curr_img, mask=curr_mask, flow=curr_flow)
        curr_img = transformed["image"]
        curr_flow = transformed['flow']
        curr_mask = transformed["mask"].unsqueeze(0)
        return dict(image=curr_img, mask=curr_mask, flow=curr_flow)

    def __len__(self):
        return len(self.total_image_paths)


def customized_collate_fn(batch, scale_list):
    scale = random.choice(scale_list)

    recombined_data = defaultdict(list)
    for item_info in batch:
        for k, v in item_info.items():
            recombined_data[k].append(v)

    results = {}
    for k, v in recombined_data.items():
        stacked_tensor = torch.stack(v, dim=0)
        if k in ['mask']:
            kwargs = dict(mode="nearest")
        else:
            kwargs = dict(mode="bilinear", align_corners=False)
        results[k] = interpolate(stacked_tensor, scale_factor=scale, recompute_scale_factor=False, **kwargs)
    return results


def customized_worker_init_fn(worker_id):
    set_seed_for_lib(arg_config['base_seed'] + worker_id)


def create_loader(data_path, training, in_size, batch_size, num_workers=0, shuffle=False, drop_last=False,
                  pin_memory=True, use_mstrain=False, use_custom_worker_init=True, get_length=False,
                  get_name_list=False) -> dict:
    if training:
        imageset = TrainDataset(root=data_path, in_size=in_size)
    else:
        imageset = TestDataset(root=data_path, in_size=in_size)
    collate_fn = customized_collate_fn if (training and use_mstrain) else None
    worker_init_fn = customized_worker_init_fn if use_custom_worker_init else None

    loader = DataLoaderX(
        dataset=imageset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn
    )
    return_info = dict(loader=loader)
    if get_length:
        return_info['length'] = len(imageset)
    if get_name_list:
        return_info['name_list'] = imageset.video_name_list
    return return_info


if __name__ == "__main__":
    imageset = TrainDataset(
        root=arg_config["usvos_data"]["tr_data_path"][0],
        in_size=arg_config["train_size"],
    )
    print(len(imageset))
