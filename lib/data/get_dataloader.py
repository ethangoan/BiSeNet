
import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

import lib.data.transform_cv2 as T
from lib.data.sampler import RepeatedDistSampler


from lib.data.cityscapes_enet import ENetCityScapes, PILToLongTensor
from lib.data.cityscapes_cv2 import CityScapes
from lib.data.coco import CocoStuff
from lib.data.ade20k import ADE20k
from lib.data.camvid import CamVid
from lib.data.customer_dataset import CustomerDataset

import torchvision.transforms as transforms
from PIL import Image



def get_data_loader(cfg, mode='train'):
    if mode == 'train':
        trans_func = T.TransformationTrain(cfg.scales, cfg.cropsize)
        batchsize = cfg.ims_per_gpu
        annpath = cfg.train_im_anns
        shuffle = True
        drop_last = True
    elif mode == 'val' or mode == 'test':
        trans_func = T.TransformationVal()
        batchsize = cfg.eval_ims_per_gpu
        annpath = cfg.val_im_anns
        shuffle = False
        drop_last = False

    if cfg.dataset == 'CityScapes':
        # for cityscapes want to include the model type  as this affects how the
        # mean and variance are used
        ds = CityScapes(cfg.im_root, annpath, trans_func=trans_func, mode=mode, model_type=cfg.model_type)
    elif cfg.dataset == 'ENetCityScapes':
        image_transform = transforms.Compose(
            [transforms.Resize((cfg.cropsize[0], cfg.cropsize[1])),
             transforms.ToTensor()])

        label_transform = transforms.Compose([
            transforms.Resize((cfg.cropsize[0], cfg.cropsize[1]), Image.NEAREST),
            PILToLongTensor()
            ])
        ds = ENetCityScapes(cfg.im_root, transform=image_transform, label_transform=label_transform,  mode=mode)
    else:
        # for cityscapes want to include the model type  as this affects how the
        # mean and variance are used
        ds = CityScapes(cfg.im_root, annpath, trans_func=trans_func, mode=mode, model_type=cfg.model_type)

        ds = eval(cfg.dataset)(cfg.im_root, annpath, trans_func=trans_func, mode=mode)

    if dist.is_initialized():
        assert dist.is_available(), "dist should be initialzed"
        if mode == 'train':
            assert not cfg.max_iter is None
            n_train_imgs = cfg.ims_per_gpu * dist.get_world_size() * cfg.max_iter
            sampler = RepeatedDistSampler(ds, n_train_imgs, shuffle=shuffle)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                ds, shuffle=shuffle)
        batchsampler = torch.utils.data.sampler.BatchSampler(
            sampler, batchsize, drop_last=drop_last
        )
        dl = DataLoader(
            ds,
            batch_sampler=batchsampler,
            num_workers=8,
            pin_memory=True,
        )
    else:
        dl = DataLoader(
            ds,
            batch_size=batchsize,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=8,
            pin_memory=True,
        )
    return dl
