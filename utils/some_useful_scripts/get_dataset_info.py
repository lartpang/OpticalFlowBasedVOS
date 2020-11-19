import os
from PIL import Image
import numpy as np
from collections import Counter
from tqdm import tqdm

object_list = []
# dataset_root = "/home/lart/Datasets/VideoSegDataset/PersonVideoSeg/Annotations"
dataset_root = "/home/lart/Datasets/VideoSegDataset/YoutubeVOS-2018/train/Annotations"
for video_name in tqdm(os.listdir(dataset_root), total=len(os.listdir(dataset_root))):
    video_path = os.path.join(dataset_root, video_name)
    first_frame_per_video_name = sorted(os.listdir(video_path))[0]
    first_frame_per_video_path = os.path.join(video_path, first_frame_per_video_name)
    first_frame_per_video = Image.open(first_frame_per_video_path)
    first_frame_per_video = np.asarray(first_frame_per_video)
    object_list.append(len(np.unique(first_frame_per_video)))

object_statistics = Counter(object_list)
print(object_statistics)
# AliTIanchi: include background, we can get the Counter({2: 455, 3: 253, 4: 81, 5: 37, 6: 24})
# Youtube 2018: include background, we can get the  Counter({2: 1940, 3: 926, 4: 522, 5: 32, 1: 31,
# 6: 20})
