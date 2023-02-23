#!/usr/bin/env python3

#!/usr/bin/env python3

cfg = dict(
    model_type='pplitesegb',
    model_config='bayes',
    var_step = 2,
    momentum=0.9,
    n_cats=19,
    num_aux_heads=4,
    lr_start=1e-5,
    weight_decay=1e-4,
    warmup_iters=100,
    max_iter=200,
    dataset='CityScapes',
    im_root='/home/ethan/tmp_data/cityscapes',
    train_im_anns='./datasets/cityscapes/train.txt',
    val_im_anns='./datasets/cityscapes/val.txt',
    scales=[0.25, 2.],
    cropsize=[512, 1024],
    eval_crop=[1024, 1024],
    eval_scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    ims_per_gpu=16,
    eval_ims_per_gpu=1,
    use_fp16=True,
    use_sync_bn=True,
    respth='./res',
)
