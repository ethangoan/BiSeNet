import torch
from torch import nn

import numpy as np

import time


class ADFSoftmax(nn.Module):
    def __init__(self, dim=1, min_variance=1e-4):
        super(ADFSoftmax, self).__init__()
        self.dim = dim
        self.min_variance = min_variance

    def keep_variance(self, x):
        return torch.clamp(x, min=self.min_variance, max=None)
        # return x + self.min_variance


    # def forward(self, features_mean, features_variance, eps=1e-5):
    #     """Softmax function applied to a multivariate Gaussian distribution.
    #     It works under the assumption that features_mean and features_variance
    #     are the parameters of a the indepent gaussians that contribute to the
    #     multivariate gaussian.
    #     Mean and variance of the log-normal distribution are computed following
    #     https://en.wikipedia.org/wiki/Log-normal_distribution."""
    #     # make sure the minumum variance is present
    #     features_variance = self.keep_variance(features_variance)
    #     log_gaussian_mean = features_mean + 0.5 * features_variance
    #     log_gaussian_variance = 2 * log_gaussian_mean

    #     log_gaussian_mean = torch.exp(log_gaussian_mean)
    #     log_gaussian_variance = torch.exp(log_gaussian_variance)
    #     log_gaussian_variance = log_gaussian_variance*(torch.exp(features_variance)-1)

    #     constant = torch.sum(log_gaussian_mean, dim=self.dim) + eps
    #     constant = constant.unsqueeze(self.dim)
    #     outputs_mean = log_gaussian_mean/constant
    #     outputs_variance = log_gaussian_variance/(constant**2)

    #     # again make sure output variance is kept
    #     outputs_variance = self.keep_variance(outputs_variance)
    #     return outputs_mean, outputs_variance

    def forward(self, features_mean, features_variance, eps=1e-5):
        """Softmax function applied to a multivariate Gaussian distribution.
        It works under the assumption that features_mean and features_variance
        are the parameters of a the indepent gaussians that contribute to the
        multivariate gaussian.
        Mean and variance of the log-normal distribution are computed following
        https://en.wikipedia.org/wiki/Log-normal_distribution."""
        # make sure the minumum variance is present
        # import time
        # time.sleep(10)
        # print(features_variance)
        features_variance = self.keep_variance(features_variance)
        # print(np.sum(features_variance.cpu().numpy() < 0))
        log_gaussian_mean = features_mean + 0.5 * features_variance
        log_gaussian_variance = 2 * log_gaussian_mean
        # print('max input var', torch.max(features_variance))
        # print('mean input var', torch.mean(features_variance))
        # time.sleep(10)
        log_gaussian_mean = torch.exp(log_gaussian_mean)
        log_gaussian_variance = torch.exp(log_gaussian_variance)
        log_gaussian_variance = log_gaussian_variance*(torch.exp(features_variance)-1)
        # now find the mean and variance of the denominator of the softmax
        denominator_mean = torch.sum(log_gaussian_mean, dim=self.dim, keepdim=True)
        denominator_variance = torch.sum(log_gaussian_variance, dim=self.dim, keepdim=True)
        # now approximate the mean and variance of this using taylor expansion approx.
        outputs_mean = log_gaussian_mean / (denominator_mean + eps)
        # print(denominator_variance)
        # print(denominator_variance.shape)
        # print(log_gaussian_variance)
        # time.sleep(10 )
        # print(np.sum(log_gaussian_variance.cpu().numpy() < 0))
        # print(np.sum(denominator_variance.cpu().numpy() < 0))
        # print(np.sum(( log_gaussian_variance + denominator_variance).cpu().numpy() < 0))
        # time.sleep(10 )
        outputs_variance = outputs_mean * torch.sqrt(log_gaussian_variance + denominator_variance + eps)
        # constant = torch.sum(log_gaussian_mean, dim=self.dim) + eps
        # constant = constant.unsqueeze(self.dim)
        # outputs_mean = log_gaussian_mean/constant
        # outputs_variance = log_gaussian_variance/(constant**2)
        # again make sure output variance is kept
        outputs_variance = self.keep_variance(outputs_variance)
        return outputs_mean, outputs_variance
