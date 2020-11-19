import os

from configs import davis16

proj_root = os.path.dirname(__file__)

_vos_data = dict(
    tr=[
        ('davis16', davis16.DAVIS16_paths)
    ],
    val=('davis16', davis16.DAVIS16_paths),
    te=[
        ('davis16', davis16.DAVIS16_paths),
    ]
)

arg_config = dict(
    # 常用配置
    resume=False,  # 是否需要恢复模型
    info="",
    data=_vos_data,
    model="BasicSiameseModel",
    model_cfg=dict(
        freeze_bn=False,
        backbone='resnet',
    ),
    use_cudnn_benchmark=False,
    has_val=True,
    has_test=True,
    use_amp=True,
    use_tta=False,
    use_mstrain=False,
    grad_max_norm=0,  # 0表示不剪裁梯度
    in_size=dict(
        tr=dict(hw=384, extra_scales=[0.5, 1.25]),
        val=dict(hw=384),
        te=dict(hw=384),
    ),
    clip_range=(0, 1),
    epoch_num=40,  # 训练周期, 0: directly test model
    batch_size=8,  # 要是继续训练, 最好使用相同的batchsize
    num_workers=4,  # 不要太大, 不然运行多个程序同时训练的时候, 会造成数据读入速度受影响
    tb_update=50,  # >0 则使用tensorboard
    print_freq=50,  # >0, 保存迭代过程中的信息
    base_seed=0,
    use_custom_worker_init=True,
)

loss_config = dict(
    bce=True,
    hel=True,
    iou=False,
    weighted_iou=False,
    mae=False,
    mse=False,
    ssim=False,
)

backbone_config = dict(
    resnet=dict(depth=101),
    deeplabv3=dict(depth=101, use_dilation=(False, False, False))
)

optimizer_config = dict(
    lr=0.004,
    strategy="trick",  # 'finetune, sgd_trick
    optimizer='sgd',
    sgd=dict(
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=False,
    ),
    adamw=dict(
        weight_decay=5e-4,
        eps=1e-8,
    ),
)

schedule_config = dict(
    sche_usebatch=True,
    lr_strategy="poly",
    clr=dict(min_lr=0.001, max_lr=0.01, step_size=2000, mode="exp_range"),
    linearonclr=dict(),
    cos=dict(warmup_length=1, min_coef=0.025, max_coef=1),
    poly=dict(warmup_length=1, lr_decay=0.9, min_coef=0.025),
    step=dict(milestones=[30, 45, 55], gamma=0.1)
)
