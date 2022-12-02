import sys

sys.path.insert(0, '.')
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import scipy

import lib.data.transform_cv2 as T
from lib.models import model_factory
from configs import set_cfg_from_file


from collections import namedtuple


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

all_labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

labels = []
class_idx = 0
for label in all_labels:
    if label.trainId     == class_idx:
        labels.append(label)
        class_idx += 1
    if class_idx >= 19:
        break

# uncomment the following line if you want to reduce cpu usage, see issue #231
#  torch.set_num_threads(4)

torch.set_grad_enabled(False)
np.random.seed(123)


class ADFSoftmax(nn.Module):
    def __init__(self, dim=1, keep_variance_fn=None):
        super(ADFSoftmax, self).__init__()
        self.dim = dim
        self._keep_variance_fn = keep_variance_fn

    def forward(self, features_mean, features_variance, eps=1e-5):
        """Softmax function applied to a multivariate Gaussian distribution.
        It works under the assumption that features_mean and features_variance
        are the parameters of a the indepent gaussians that contribute to the
        multivariate gaussian.
        Mean and variance of the log-normal distribution are computed following
        https://en.wikipedia.org/wiki/Log-normal_distribution."""
        log_gaussian_mean = features_mean + 0.5 * features_variance
        log_gaussian_variance = 2 * log_gaussian_mean

        log_gaussian_mean = torch.exp(log_gaussian_mean)
        log_gaussian_variance = torch.exp(log_gaussian_variance)
        log_gaussian_variance = log_gaussian_variance*(torch.exp(features_variance)-1)

        constant = torch.sum(log_gaussian_mean, dim=self.dim) + eps
        constant = constant.unsqueeze(self.dim)
        outputs_mean = log_gaussian_mean/constant
        outputs_variance = log_gaussian_variance/(constant**2)

        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance




# following fns and color maps taken from CSVAIL
# https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/mit_semseg/utils.py
def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def colorEncode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb
colors = scipy.io.loadmat('color150.mat')['colors']


# args
parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str, default='configs/bisenetv2.py',)
parse.add_argument('--weight-path', type=str, default='./res/model_final.pth',)
parse.add_argument('--img-path', dest='img_path', type=str, default='./example.png',)
args = parse.parse_args()
cfg = set_cfg_from_file(args.config)


palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

# define model
net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval_bayes')
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
weight_var = torch.load('weight_var.pt')
bias_var = torch.load('bias_var.pt')

net.head.conv_var.weight.data = weight_var
net.head.conv_var.bias.data = bias_var
print(net)
net.eval()
net.cuda()

# prepare data
to_tensor = T.ToTensor(
    mean=(0.3257, 0.3690, 0.3223), # city, rgb
    std=(0.2112, 0.2148, 0.2115),
)
im = cv2.imread(args.img_path)[:, :, ::-1]
im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()

# shape divisor
org_size = im.size()[2:]
new_size = [math.ceil(el / 64) * 32 for el in im.size()[2:]]
print('new size', new_size)

# inference
im = F.interpolate(im, size=new_size, align_corners=False, mode='bilinear')
print(im.shape)
out_mean = 0.0
out_entropy = 0.0
# T = 100
# print(new_size)
# for i in range(T):
#     print(i)
#     out = net(im)[0]
#     out = F.interpolate(out, size=org_size, align_corners=False, mode='bilinear')
#     out = F.softmax(out)
#     out_mean += out
#     out_entropy += -(out * torch.log(out)).mean(1)
#     print(out_entropy.shape)

# out_mean = out_mean / T
# out_entropy = out_entropy / T

# # visualize
# out_mean = out_mean.argmax(1).squeeze().detach().cpu().numpy()
# print(out_mean.shape)
# out_entropy = out_entropy.squeeze().detach().cpu().numpy()
# print(out_mean)
# print(org_size)
# # pred = palette[out_mean.argmax(1)]


# pred_col = colorEncode(out_mean, colors)
# print(out_entropy)
# cv2.imwrite('./res.jpg', pred_col)
# cv2.imwrite('./res_entropy.jpg', out_entropy / np.max(out_entropy))

import matplotlib.pyplot as plt
# plt.figure()
# plt.imshow(out_entropy / np.max(out_entropy))
# plt.savefig('ent.png')

# print(net.head.conv_bayes.rho_kernel)


# net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval_bayes')
# net.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
# net.eval()
# net.cuda()

# print(net)
# weight_var = torch.load('weight_var.pt')
# bias_var = torch.load('bias_var.pt')
# now do a forward pass
logit_mean, logit_var = net(im)
for i in range(19):
    plt.figure()
    out_var_class = logit_var[0, i, :, :]
    plt.imshow(torch.squeeze(out_var_class / torch.max(out_var_class)).cpu().numpy())
    plt.title(labels[i].name)
    plt.savefig(f'var_{i}.png')
    plt.clf()
    out_class = F.softmax(logit_mean)[0, i, :, :].cpu().numpy()
    plt.imshow(np.squeeze(out_class))
    plt.title(labels[i].name)
    plt.savefig(f'logit/im_{i}.png')





# out_mean, feat_basis = net(im)

# test = F.conv2d(torch.square(feat_basis), weight_var, bias=bias_var)
# print(test.shape)
# out_var = F.interpolate(test, size=org_size, align_corners=False, mode='bilinear')

# for i in range(19):
#     plt.figure()
#     out_var_class = out_var[0, i, :, :]
#     plt.imshow(torch.squeeze(out_var_class / torch.max(out_var_class)).cpu().numpy())
#     plt.title(labels[i].name)
#     plt.savefig(f'var_{i}.png')
#     plt.clf()
#     out_class = F.softmax(out_mean)[0, i, :, :].cpu().numpy()
#     plt.imshow(np.squeeze(out_class))
#     plt.title(labels[i].name)
#     plt.savefig(f'pred/im_{i}.png')


out = F.softmax(torch.Tensor(logit_mean))
out_entropy = -(out * torch.log(out)).mean(1)
# visualize
out = out.argmax(1).squeeze().detach().cpu().numpy()
out_entropy = out_entropy.squeeze().detach().cpu().numpy()
# pred = palette[out_mean.argmax(1)]
print(out.shape)

pred_col = colorEncode(out, colors)
print(out_entropy)
cv2.imwrite('./res.jpg', pred_col)
cv2.imwrite('./res_entropy.jpg', out_entropy / np.max(out_entropy))

print(out)

# now have a look at the uncertainty propegation stuff
adf_softmax = ADFSoftmax()
s_mean, s_var = adf_softmax(logit_mean, logit_var)

for i in range(19):
    plt.figure()
    s_var_class = s_var[0, i, :, :]
    plt.imshow(torch.squeeze(s_var_class ).cpu().numpy())
    plt.title(labels[i].name)
    plt.savefig(f'softmax/var_{i}.png')


# now try and compute the entropy in the Gaussian side of things
gaussian_entropy = 0.5 * torch.log((torch.prod(logit_var.double(), 1))).squeeze().cpu().numpy()
print(np.max(gaussian_entropy))
print(np.min(gaussian_entropy))
plt.figure()
plt.imshow(gaussian_entropy)
plt.title('gaussian_entropy')
plt.savefig(f'gaussian_entropy.png')

plt.figure()
plt.imshow(out_entropy)
plt.title('softmax_entropy')
plt.savefig(f'softmax_entropy.png')
