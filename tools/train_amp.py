#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys

sys.path.insert(0, '.')
import os
import os.path as osp
import random
import logging
import time
import argparse
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.cuda.amp as amp

from lib.models import model_factory
from configs import set_cfg_from_file
from lib.data import get_data_loader
from evaluate import eval_model
from lib.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler, WarmupOnlyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg

from bayesian_torch.models.dnn_to_bnn import get_kl_loss

torch.autograd.set_detect_anomaly(True)
## fix all random seeds
#  torch.manual_seed(123)
#  torch.cuda.manual_seed(123)
#  np.random.seed(123)
#  random.seed(123)
#  torch.backends.cudnn.deterministic = True
#  torch.backends.cudnn.benchmark = True
#  torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        '--config',
        dest='config',
        type=str,
        default='configs/bisenetv2.py',
    )
    parse.add_argument(
        '--finetune-from',
        type=str,
        default=None,
    )
    parse.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parse.add_argument('-g',
                       '--gpus',
                       default=1,
                       type=int,
                       help='number of gpus per node')
    parse.add_argument('-nr',
                       '--nr',
                       default=0,
                       type=int,
                       help='ranking within the nodes')
    args = parse.parse_args()
    args.world_size = args.gpus * args.nodes
    return args


args = parse_args()
cfg = set_cfg_from_file(args.config)


def set_model(lb_ignore=255):
    logger = logging.getLogger()
    net = model_factory[cfg.model_type](cfg.n_cats)
    if not args.finetune_from is None:
        logger.info(f'load pretrained weights from {args.finetune_from}')
        net.load_state_dict(torch.load(args.finetune_from, map_location='cpu'))
    if cfg.use_sync_bn: net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    # if the model contains some bayesian references, need to set the prior terms
    # print(net.head.conv_bayes.prior_bias_sigma)
    # print(net.head.conv_bayes.rho_bias)
    if 'bayes' in cfg.model_type:
        pass
        # net.head.conv_bayes.prior_weight_mu.data = net.head.conv_bayes.mu_kernel        # )
        # net.head.conv_bayes.prior_weight_sigma.data = torch.log1p(
        #     torch.exp(net.head.conv_bayes.rho_kernel))
        # net.head.conv_bayes.prior_bias_mu.data = net.head.conv_bayes.mu_bias
        # net.head.conv_bayes.prior_bias_sigma.data = torch.log1p(
        #     torch.exp(net.head.conv_bayes.rho_bias))
        # print(net.head.conv_bayes.prior_bias_sigma.device)
        # print(net.head.conv_bayes.prior_bias_mu.device)
        # print(net.head.conv_bayes.prior_weight_sigma.device)
        # print(net.head.conv_bayes.prior_weight_mu.device)
        # now turn the gradients off
        # net.head.conv_bayes.prior_weight_mu.requires_grad = False
        # net.head.conv_bayes.prior_weight_sigma.requires_grad = False
        # net.head.conv_bayes.prior_bias_mu.requires_grad = False
        # net.head.conv_bayes.prior_bias_sigma.requires_grad = False
    net.cuda()
    net.train()
    criteria_pre = OhemCELoss(0.7, lb_ignore)
    criteria_aux = [
        OhemCELoss(0.7, lb_ignore) for _ in range(cfg.num_aux_heads)
    ]
    return net, criteria_pre, criteria_aux


def set_optimizer(model):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params(
        )
        #  wd_val = cfg.weight_decay
        no_wd_val = 0.0
        params_list = [
            {
                'params': wd_params,
                # 'weight_decay': wd_val
            },
            {
                'params': nowd_params,
                'weight_decay': no_wd_val
            },
            {
                'params': lr_mul_wd_params,
                # 'weight_decay': wd_val,
                'lr': cfg.lr_start * 10
            },
            {
                'params': lr_mul_nowd_params,
                'weight_decay': no_wd_val,
                'lr': cfg.lr_start * 10
            },
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {
                'params': wd_params,
            },
            {
                'params': non_wd_params,
                'weight_decay': 0
            },
        ]

    optim = torch.optim.SGD(
        params_list,
        lr=cfg.lr_start,
        momentum=0.0,
        weight_decay=cfg.weight_decay,
    )

    return optim


def set_model_dist(net):
    local_rank = int(os.environ['LOCAL_RANK'])
    net = nn.parallel.DistributedDataParallel(
        net,
        device_ids=[
            local_rank,
        ],
        #  find_unused_parameters=True,
        output_device=local_rank)
    return net


def set_meters():
    time_meter = TimeMeter(cfg.max_iter)
    loss_meter = AvgMeter('loss')
    kl_meter = AvgMeter('kl')
    loss_pre_meter = AvgMeter('loss_prem')
    loss_aux_meters = [
        AvgMeter('loss_aux{}'.format(i)) for i in range(cfg.num_aux_heads)
    ]
    return time_meter, loss_meter, loss_pre_meter, kl_meter, loss_aux_meters


def nan_hook(self, inp, output):
    if not isinstance(output, tuple):
        outputs = [output]
    else:
        outputs = output

    for i, out in enumerate(outputs):
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            print("In", self.__class__.__name__)
            raise RuntimeError(
                f"Found NAN in output {i} at indices: ", nan_mask.nonzero(),
                "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])




def kl_div(mu_q, sigma_q, mu_p, sigma_p):
    """Calculates kl divergence between two gaussians (Q || P)
         Parameters:
              * mu_q: torch.Tensor -> mu parameter of distribution Q
              * sigma_q: torch.Tensor -> sigma parameter of distribution Q
              * mu_p: float -> mu parameter of distribution P
              * sigma_p: float -> sigma parameter of distribution P
         returns torch.Tensor of shape 0
         """
    kl = torch.log(sigma_p) - torch.log(
        sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 * (sigma_p**2)) - 0.5
    return kl.mean()


def kl_loss(mu_kernel, rho_kernel, mu_bias, rho_bias, prior_weight_mu, prior_weight_sigma, prior_bias_mu, prior_bias_sigma):
    sigma_weight = torch.log1p(torch.exp(rho_kernel))
    print(sigma_weight.device)
    kl = kl_div(mu_kernel, sigma_weight, prior_weight_mu, prior_weight_sigma)
    print(f'kl weight {kl.device}')
    sigma_bias = torch.log1p(torch.exp(rho_bias))
    print(f'kl bias {kl.device}')
    kl += kl_div(mu_bias, sigma_bias, prior_bias_mu, prior_bias_sigma)
    return kl

def train():
    logger = logging.getLogger()

    ## dataset
    dl = get_data_loader(cfg, mode='train')

    ## model

    net, criteria_pre, criteria_aux = set_model(dl.dataset.lb_ignore)
    # for submodule in net.modules():
    #     print(submodule)
    # print(net)

    # for submodule in net.modules():
    #     submodule.register_forward_hook(nan_hook)

    # print(net.head.conv_bayes.mu_kernel.device)
    # print(net.head.conv_bayes.rho_kernel.device)
    # print(net.head.conv_bayes.mu_bias.device)
    # print(net.head.conv_bayes.rho_bias.device)
    # for param in net.parameters():
    #     param.requires_grad = False
    # net.head.conv_bayes.mu_kernel.requires_grad = True
    # net.head.conv_bayes.mu_bias.requires_grad = True
    # net.head.conv_bayes.rho_kernel.requires_grad = True
    # net.head.conv_bayes.rho_bias.requires_grad = True

    # net.head.conv_out.weight.requires_grad = True
    # net.head.conv_out.bias.requires_grad = True

    # print('params')
    # print(net.head.conv_bayes.mu_kernel.device)
    # print(net.head.conv_bayes.rho_kernel.device)
    # print(net.head.conv_bayes.mu_bias.device)
    # print(net.head.conv_bayes.rho_bias.device)

    # print('priors')
    # print(net.head.conv_bayes.prior_bias_mu.device)
    # print(net.head.conv_bayes.prior_bias_sigma.device)
    # print(net.head.conv_bayes.prior_weight_mu.device)
    # print(net.head.conv_bayes.prior_weight_sigma.device)
    # # for param in net.parameters():
    # #     print(torch.sum(torch.isnan(param)))


    # print(net.head.conv_bayes.mu_bias)
    ## optimizer
    optim = set_optimizer(net)

    ## mixed precision training
    scaler = amp.GradScaler()

    ## ddp training
    net = net.cuda()
    net = set_model_dist(net)
    # net = net.cuda()
    print(net)
    ## meters
    time_meter, loss_meter, kl_meter, loss_pre_meter, loss_aux_meters = set_meters()

    ## lr scheduler
    # lr_schdr = WarmupPolyLrScheduler(
    #     optim,
    #     power=0.9,
    #     max_iter=cfg.max_iter,
    #     warmup_iter=cfg.warmup_iters,
    #     warmup_ratio=0.1,
    #     warmup='exp',
    #     last_epoch=-1,
    # )

    lr_schdr = WarmupOnlyLrScheduler(
        optim,
        power=0.9,
        max_iter=cfg.max_iter,
        warmup_iter=cfg.warmup_iters,
        warmup_ratio=0.1,
        warmup='exp',
        last_epoch=-1,
    )


    if 'bisenetv2' in cfg.model_type:
        weight_sum = torch.zeros(net.module.head.conv_out.weight.shape).cuda()
        bias_sum = torch.zeros(net.module.head.conv_out.bias.shape).cuda()
        weight_squared_sum = torch.zeros(net.module.head.conv_out.weight.shape).cuda()
        bias_squared_sum = torch.zeros(net.module.head.conv_out.bias.shape).cuda()
    else:
        # is enet
        weight_sum = torch.zeros(net.module.fullconv.weight.shape).cuda()
        weight_squared_sum = torch.zeros(net.module.fullconv.weight.shape).cuda()


    # print(net)
    # print(torch.log1p(net.module.head.conv_bayes.prior_weight_sigma))
    ## train loop
    for it, (im, lb) in enumerate(dl):
        im = im.cuda()
        lb = lb.cuda()

        lb = torch.squeeze(lb, 1)

        optim.zero_grad()

        with amp.autocast(enabled=cfg.use_fp16):
            if 'bayes' in cfg.model_type:
                logits, kl, *logits_aux = net(im)
                loss_pre = criteria_pre(logits, lb)
                loss_aux = [
                    crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux)
                ]

                # kl = net.module.head.conv_bayes.kl_loss()
                # kl = kl_loss(net.module.head.conv_bayes.mu_kernel,
                #              net.module.head.conv_bayes.rho_kernel,
                #              net.module.head.conv_bayes.mu_bias,
                #              net.module.head.conv_bayes.rho_bias,
                #              net.module.head.conv_bayes.prior_weight_mu,
                #              net.module.head.conv_bayes.prior_weight_sigma,
                #              net.module.head.conv_bayes.prior_bias_mu,
                #              net.module.head.conv_bayes.prior_bias_sigma)
                # print(f'kl device {kl.device}')
                # print(f'loss pre {loss_pre.device}')
                # TODO include number of gpus here
                loss = loss_pre + kl / cfg.ims_per_gpu # + sum(loss_aux)
                kl_item = kl.item()
            elif 'enet' in cfg.model_type:
                logits  = net(im)
                loss_pre = criteria_pre(logits, lb)
                scaler.scale(loss_pre).backward()
                kl_item = 0
                loss = loss_pre
                # loss aux is set just as a dummy varible
                # it isn't actually tracking anything important
                # or tracking anything really
                loss_aux = [
                    crit(0, 0) for crit in criteria_aux
                ]
            # is normal BiSeNet
            else:
                logits, *logits_aux = net(im)
                loss_pre = criteria_pre(logits, lb)
                loss_aux = [
                    crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux)
                ]

                loss = loss_pre# + sum(loss_aux)
                kl_item = 0
                scaler.scale(loss).backward()
        scaler.step(optim)
        if (it >= cfg.warmup_iters) and ('bayes' in cfg.model_type):
            # now update our sum parameters for the  weight and bias terms
            weight_sum += net.module.head.conv_out.weight
            bias_sum += net.module.head.conv_out.bias
            weight_squared_sum += torch.square(net.module.head.conv_out.weight)
            bias_squared_sum += torch.square(net.module.head.conv_out.bias)


        # for param in net.parameters():
        #   print("param.data",torch.isfinite(param.data).all())
        #   print("param.grad.data",torch.isfinite(param.grad.data).all(),"\n")
        scaler.update()
        torch.cuda.synchronize()

        # print(net.head.conv_bayes.prior_bias_sigma)
        time_meter.update()
        loss_meter.update(loss.item())
        loss_pre_meter.update(loss_pre.item())
        if 'bayes' in cfg.model_type:
            kl_meter.update(kl_item)
        if 'bisenet' in cfg.model_type:
            _ = [
                mter.update(lss.item())
                for mter, lss in zip(loss_aux_meters, loss_aux)
                ]
        else:
           kl_meter = None
           loss_aux_meters = None


        ## print training log message
        if (it + 1) % 100 == 0:
            lr = lr_schdr.get_lr()
            lr = sum(lr) / len(lr)
            print_log_msg(it, cfg.max_iter, lr, time_meter, loss_meter,
                          kl_meter, loss_pre_meter, loss_aux_meters)
        lr_schdr.step()

    if 'bayes' in cfg.model_type:
        # divide the sum terms by the number of iterations to get the value for the SWAG params
        weight_mean = weight_sum / (cfg.max_iter - cfg.warmup_iters)
        bias_mean = bias_sum / (cfg.max_iter - cfg.warmup_iters)
        weight_squared_sum = weight_squared_sum / (cfg.max_iter - cfg.warmup_iters)
        bias_squared_sum = bias_squared_sum / (cfg.max_iter - cfg.warmup_iters)
        weight_var = weight_squared_sum - torch.square(weight_mean)
        bias_var = bias_squared_sum - torch.square(bias_mean)

        torch.save(weight_mean, 'weight_mean.pt')
        torch.save(bias_mean, 'bias_mean.pt')
        torch.save(weight_var, 'weight_var.pt')
        torch.save(bias_var, 'bias_var.pt')
        print('weight_var = ', weight_var)
        print('bias_var = ', bias_var)

    ## dump the final model and evaluate the result
    save_pth = osp.join(cfg.respth, 'model_final.pth')
    logger.info('\nsave models to {}'.format(save_pth))
    state = net.module.state_dict()
    if dist.get_rank() == 0: torch.save(state, save_pth)


    # net.module.head.conv_out.weight.data =  weight_mean
    logger.info('\nevaluating the final model')
    torch.cuda.empty_cache()
    iou_heads, iou_content, f1_heads, f1_content = eval_model(cfg, net.module)
    logger.info('\neval results of f1 score metric:')
    logger.info('\n' +
                tabulate(f1_content, headers=f1_heads, tablefmt='orgtbl'))
    logger.info('\neval results of miou metric:')
    logger.info('\n' +
                tabulate(iou_content, headers=iou_heads, tablefmt='orgtbl'))

    return


def main():
    # os.environ['MASTER_ADDR'] = '192.168.1.3'
    # os.environ['MASTER_PORT'] = '8888'
    # os.environ['LOCAL_RANK'] = 1
    local_rank = int(os.environ['LOCAL_RANK'])
    print(f'local rank {local_rank}')
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    # dist.init_process_group(
    #     backend='nccl',
    #     init_method='env://',
    #     world_size=args.world_size,
    #     rank=rank
    # )
    if not osp.exists(cfg.respth): os.makedirs(cfg.respth)
    setup_logger(f'{cfg.model_type}-{cfg.dataset.lower()}-train', cfg.respth)
    train()


if __name__ == "__main__":
    main()
