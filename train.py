import copy
import os
from collections import defaultdict
from datetime import datetime
from pprint import pprint

import cv2
import numpy as np
import torch
import ttach as tta
from torch.cuda import amp
from tqdm import tqdm

import network as network_lib
from config import arg_config, backbone_config, loss_config, optimizer_config, proj_root, schedule_config
from data.dataloader import create_loader
from data.utils import read_binary_array
from loss import get_loss_combination_with_cfg
from metrics import f_boundary, jaccard
from utils.misc import (
    construct_exp_name,
    construct_path,
    construct_print,
    initialize_seed_cudnn,
    make_log,
    pre_copy,
    pre_mkdir,
    pretty_print,
)
from utils.recorder import AvgMeter, CustomizedTimer, TBRecorder
from utils.tools_funcs import (
    clip_to_normalize, make_optimizer_with_cfg, make_scheduler_with_cfg, resume_checkpoint, save_checkpoint,
)

DEVICES = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def tta_aug(data, model, transforms):
    if transforms is None:
        model_output = model(data=dict(curr_jpeg=data.get('curr_jpeg', None),
                                       curr_flow=data.get('curr_flow', None)))
        outputs = model_output['curr_seg']
    else:
        curr_jpegs = data.get('curr_jpeg', None)
        curr_flows = data.get('curr_flow', None)
        auged_outputs = []
        for transformer in transforms:
            # augment image
            if curr_jpegs is not None:
                augmented_jpegs = transformer.augment_image(curr_jpegs)
            else:
                augmented_jpegs = None
            if curr_flows is not None:
                augmented_flows = transformer.augment_image(curr_flows)
            else:
                augmented_flows = None
            # pass to model
            model_output = model(data=dict(curr_jpeg=augmented_jpegs,
                                           curr_flow=augmented_flows))
            inter_outputs = model_output['curr_seg']
            # reverse augmentation
            deaug_logits = transformer.deaugment_mask(inter_outputs)
            # save results
            auged_outputs.append(deaug_logits)
        # reduce results as you want, e.g mean/max/min
        outputs = torch.mean(torch.stack(auged_outputs, dim=0), dim=0)
    return outputs


@CustomizedTimer(cus_msg="Test")
def test(model, data_loader, save_path=""):
    """
    为了计算方便，训练过程中的验证与测试都直接计算指标J和F，不再先生成再输出，
    所以这里的指标仅作一个相对的参考，具体真实指标需要使用测试代码处理
    """
    model.eval()
    tqdm_iter = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)

    if arg_config['use_tta']:
        construct_print("We will use Test Time Augmentation!")
        transforms = tta.Compose([  # 2*3
            tta.HorizontalFlip(),
            tta.Scale(scales=[0.75, 1, 1.5], interpolation='bilinear', align_corners=False)
        ])
    else:
        transforms = None

    results = defaultdict(list)
    for test_batch_id, test_data in tqdm_iter:
        tqdm_iter.set_description(f"te=>{test_batch_id + 1}")

        with torch.no_grad():
            curr_jpegs = test_data["image"].to(DEVICES, non_blocking=True)
            curr_flows = test_data["flow"].to(DEVICES, non_blocking=True)
            preds_logits = tta_aug(model=model, transforms=transforms,
                                   data=dict(curr_jpeg=curr_jpegs, curr_flow=curr_flows))
            preds_prob = preds_logits.sigmoid().squeeze().cpu().detach()  # float32

        for i, pred_prob in enumerate(preds_prob.numpy()):
            curr_mask_path = test_data["mask_path"][i]
            video_name, mask_name = curr_mask_path.split(os.sep)[-2:]
            mask = read_binary_array(curr_mask_path, thr=0)
            mask_h, mask_w = mask.shape

            pred_prob = cv2.resize(pred_prob, dsize=(mask_w, mask_h), interpolation=cv2.INTER_LINEAR)
            pred_prob = clip_to_normalize(data_array=pred_prob, clip_range=arg_config["clip_range"])
            pred_seg = np.where(pred_prob > 0.5, 255, 0).astype(np.uint8)

            results[video_name].append(
                (jaccard.db_eval_iou(annotation=mask, segmentation=pred_seg),
                 f_boundary.db_eval_boundary(annotation=mask, segmentation=pred_seg))
            )

            if save_path:
                pred_video_path = os.path.join(save_path, video_name)
                if not os.path.exists(pred_video_path):
                    os.makedirs(pred_video_path)
                pred_frame_path = os.path.join(pred_video_path, mask_name)
                cv2.imwrite(pred_frame_path, pred_seg)

    j_f_collection = []
    for video_name, video_scores in results.items():
        j_f_for_video = np.mean(np.array(video_scores), axis=0).tolist()
        results[video_name] = j_f_for_video
        j_f_collection.append(j_f_for_video)
    results['average'] = np.mean(np.array(j_f_collection), axis=0).tolist()
    return pretty_print(results)


def cal_total_loss(seg_logits, seg_gts):
    loss_list = []
    loss_str_list = []

    for loss_name, loss_func in loss_funcs.items():
        loss = loss_func(seg_logits=seg_logits, seg_gts=seg_gts)
        loss_list.append(loss)
        loss_str_list.append(f"{loss_name}: {loss.item():.5f}")

    train_loss = sum(loss_list)
    return dict(loss=train_loss, loss_str=loss_str_list)


@CustomizedTimer(cus_msg="Train An Epoch")
def train_epoch(data_loader, curr_epoch):
    loss_record = AvgMeter()
    construct_print(f"Exp_Name: {exp_name}")
    for curr_iter_in_epoch, data in enumerate(data_loader):
        num_iter_per_epoch = len(data_loader)
        curr_iter = curr_epoch * num_iter_per_epoch + curr_iter_in_epoch

        curr_jpegs = data["image"].to(DEVICES, non_blocking=True)
        curr_masks = data["mask"].to(DEVICES, non_blocking=True)
        curr_flows = data["flow"].to(DEVICES, non_blocking=True)
        with amp.autocast(enabled=arg_config['use_amp']):
            preds = model(data=dict(curr_jpeg=curr_jpegs, curr_flow=curr_flows))
            seg_jpeg_logits = preds["curr_seg"]
            seg_flow_logits = preds["curr_seg_flow"]

            jpeg_loss_info = cal_total_loss(seg_logits=seg_jpeg_logits, seg_gts=curr_masks)
            flow_loss_info = cal_total_loss(seg_logits=seg_flow_logits, seg_gts=curr_masks)
            total_loss = jpeg_loss_info['loss'] + flow_loss_info['loss']
            total_loss_str = jpeg_loss_info['loss_str'] + flow_loss_info['loss_str']

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        iter_loss = total_loss.item()
        batch_size = curr_jpegs.size(0)
        loss_record.update(iter_loss, batch_size)

        if (arg_config["tb_update"] > 0 and curr_iter % arg_config["tb_update"] == 0):
            tb_recorder.record_curve("loss_avg", loss_record.avg, curr_iter)
            tb_recorder.record_curve("iter_loss", iter_loss, curr_iter)
            tb_recorder.record_curve("lr", optimizer.param_groups, curr_iter)
            tb_recorder.record_image("jpegs", curr_jpegs, curr_iter)
            tb_recorder.record_image("flows", curr_flows, curr_iter)
            tb_recorder.record_image("masks", curr_masks, curr_iter)
            tb_recorder.record_image("segs", preds["curr_seg"].sigmoid(), curr_iter)

        if (arg_config["print_freq"] > 0 and curr_iter % arg_config["print_freq"] == 0):
            lr_string = ",".join([f"{x:.7f}" for x in scheduler.get_last_lr()])
            log = (
                f"[{curr_iter_in_epoch}:{num_iter_per_epoch},{curr_iter}:{num_iter},"
                f"{curr_epoch}:{end_epoch}][{list(curr_jpegs.shape)}]"
                f"[Lr:{lr_string}][M:{loss_record.avg:.5f},C:{iter_loss:.5f}]{total_loss_str}"
            )
            print(log)
            make_log(path_dict["tr_log"], log)

        if scheduler_usebatch:
            scheduler.step()


def train(tr_loader, val_loader=None):
    for curr_epoch in range(start_epoch, end_epoch):
        if val_loader is not None:
            seg_results = test(model=model, data_loader=val_loader)
            msg = f"Epoch: {curr_epoch}, Results on the valsel: {seg_results}"
            print(msg)
            make_log(path_dict["te_log"], msg)

        model.train()
        train_epoch(data_loader=tr_loader, curr_epoch=curr_epoch)

        # 根据周期修改学习率
        if not scheduler_usebatch:
            scheduler.step()

        # 每个周期都进行保存测试，保存的是针对第curr_epoch+1周期的参数
        save_checkpoint(exp_name=exp_name,
                        model=model,
                        current_epoch=curr_epoch + 1,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        full_net_path=path_dict["final_full_net"],
                        state_net_path=path_dict["final_state_net"])


construct_print(f"{datetime.now()}: Initializing...")
construct_print(f"Project Root: {proj_root}")
pprint(arg_config)

# construct exp_name
exp_name = construct_exp_name(arg_dict=arg_config, extra_dicts=[optimizer_config, schedule_config])
path_dict = construct_path(proj_root=proj_root, exp_name=exp_name)

initialize_seed_cudnn(seed=0, use_cudnn_benchmark=False \
    if arg_config["use_mstrain"] else arg_config["use_cudnn_benchmark"])
pre_mkdir(path_config=path_dict)
pre_copy(
    main_file_path=__file__,
    all_config=dict(args=arg_config, path=path_dict, opti=optimizer_config, sche=schedule_config),
    proj_root=proj_root,
)

if arg_config["tb_update"] > 0:
    tb_recorder = TBRecorder(path_dict["tb"])

tr_data_info = arg_config["data"]["tr"]
tr_loader_info = create_loader(data_path=arg_config["data"]["tr"],
                               training=True,
                               in_size=arg_config['in_size']['tr'],
                               batch_size=arg_config['batch_size'],
                               num_workers=arg_config['num_workers'],
                               shuffle=True,
                               drop_last=True,
                               pin_memory=True,
                               use_mstrain=arg_config['use_mstrain'],
                               use_custom_worker_init=arg_config['use_custom_worker_init'],
                               get_length=True)
construct_print(f"Total length of the trainset is {tr_loader_info['length']}")

start_epoch = 0
end_epoch = arg_config["epoch_num"]
num_iter = end_epoch * len(tr_loader_info['loader'])

network_realname = arg_config["model"]
if hasattr(network_lib, network_realname):
    model_cfg = copy.deepcopy(arg_config["model_cfg"])
    backbone_name = model_cfg['backbone']
    model_cfg['backbone_cfg'] = backbone_config[backbone_name]
    model = getattr(network_lib, network_realname)(model_cfg).to(DEVICES)
else:
    raise Exception("Please add the network into the __init__.py.")

loss_funcs = get_loss_combination_with_cfg(loss_cfg=loss_config)

optimizer = make_optimizer_with_cfg(model=model, optimizer_cfg=optimizer_config)

end_epoch = arg_config["epoch_num"]
num_iter = end_epoch * len(tr_loader_info['loader'])
scheduler_usebatch = schedule_config["sche_usebatch"]
scheduler = make_scheduler_with_cfg(optimizer=optimizer,
                                    total_num=num_iter if scheduler_usebatch else end_epoch,
                                    scheduler_cfg=schedule_config, )

# Creates a GradScaler once at the beginning of training.
scaler = amp.GradScaler(enabled=arg_config['use_amp'])

if arg_config["resume"]:
    start_epoch = resume_checkpoint(exp_name=exp_name,
                                    load_path=path_dict["final_full_net"],
                                    model=model,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    scaler=scaler,
                                    mode="all",
                                    force_load=True)

if start_epoch != end_epoch:
    if arg_config["has_val"]:
        val_data_name, val_data_path = arg_config["data"]["val"]
        construct_print(f"We will validate the model on {val_data_name} per epoch.")
        val_loader_info = create_loader(data_path=val_data_path,
                                        training=False,
                                        in_size=arg_config['in_size']['val'],
                                        batch_size=arg_config['batch_size'],
                                        num_workers=arg_config['num_workers'],
                                        shuffle=False,
                                        drop_last=False,
                                        pin_memory=True,
                                        use_custom_worker_init=arg_config['use_custom_worker_init'])
        train(tr_loader=tr_loader_info['loader'], val_loader=val_loader_info['loader'])
    else:
        construct_print("We will not validate the model.")
        train(tr_loader=tr_loader_info['loader'], val_loader=None)
else:
    if not arg_config["resume"]:
        resume_checkpoint(exp_name=exp_name, load_path=path_dict["final_state_net"], model=model, mode="onlynet")

if arg_config["has_test"]:
    for te_data_name, te_data_path in arg_config["data"]["te"]:
        construct_print(f"Testing with testset: {te_data_name}")
        te_loader_info = create_loader(data_path=te_data_path,
                                       training=False,
                                       in_size=arg_config['in_size']['te'],
                                       batch_size=arg_config['batch_size'],
                                       num_workers=arg_config['num_workers'],
                                       shuffle=False,
                                       drop_last=False,
                                       pin_memory=True,
                                       use_custom_worker_init=arg_config['use_custom_worker_init'])
        pred_save_path = os.path.join(path_dict["save"], te_data_name)
        seg_results = test(model=model, data_loader=te_loader_info['loader'], save_path=pred_save_path)
        msg = (f"Results on the testset({te_data_name}:'{te_data_path['root']}'):\n{seg_results}")
        print(msg)
        make_log(path_dict["te_log"], msg)

construct_print(f"{datetime.now()}: End training...")
