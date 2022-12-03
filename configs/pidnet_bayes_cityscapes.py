## pidnet-s
cfg = dict(
    model_type='pidnet',
    model_config='bayes',
    var_step = 20,
    n_cats=19,
    num_aux_heads=0,
    lr_start=2e-1,
    weight_decay=5e-4,
    momentum=0.0,
    warmup_iters=1000,
    max_iter=5000,
    dataset='CityScapes',
    im_root='/home/ethan/exp_data/cityscapes',
    train_im_anns='./datasets/cityscapes/train.txt',
    val_im_anns='./datasets/cityscapes/val.txt',
    scales=[0.25, 2.],
    cropsize=[720, 960],
    eval_crop=[720, 960],
    eval_scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    ims_per_gpu=12,
    eval_ims_per_gpu=2,
    use_fp16=True,
    use_sync_bn=True,
    respth='./res/pidnet-cityscapes',
)
