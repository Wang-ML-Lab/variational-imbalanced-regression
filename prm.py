import logging
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import calibrate_mean_var_bayes


print = logging.info


class PRM(nn.Module):
    def __init__(self, feature_dim, bucket_num=100, bucket_start=3, start_update=0, start_smooth=1,
                 kernel='gaussian', ks=5, sigma=2, momentum=0.9, bins=1, device=None):
        super(PRM, self).__init__()
        self.feature_dim = feature_dim
        self.bucket_num = bucket_num
        self.bucket_start = bucket_start
        self.kernel_window = self._get_kernel_window(kernel, ks, sigma, bins, device)
        self.half_ks = (ks - 1) // 2
        self.bins = bins
        self.half_bin = (bins - 1) // 2
        self.momentum = momentum
        self.start_update = start_update
        self.start_smooth = start_smooth

        self.device = device

        self.register_buffer('epoch', torch.zeros(1).fill_(start_update))
        self.register_buffer('running_mean', torch.cat((torch.zeros(bucket_num - bucket_start, feature_dim),
                                        torch.ones(bucket_num - bucket_start, feature_dim)), dim=1))
        self.register_buffer('running_var', torch.cat((torch.ones(bucket_num - bucket_start, feature_dim),
                                               torch.ones(bucket_num - bucket_start, feature_dim)), dim=1))
        
        self.register_buffer('running_mean_last_epoch', torch.cat((torch.zeros(bucket_num - bucket_start, feature_dim),
                                               torch.ones(bucket_num - bucket_start, feature_dim)), dim=1))
        self.register_buffer('running_var_last_epoch', torch.cat((torch.ones(bucket_num - bucket_start, feature_dim),
                                               torch.ones(bucket_num - bucket_start, feature_dim)), dim=1))
        
        self.register_buffer('smoothed_mean_last_epoch', torch.cat((torch.zeros(bucket_num - bucket_start, feature_dim),
                                               torch.ones(bucket_num - bucket_start, feature_dim)), dim=1))
        self.register_buffer('smoothed_var_last_epoch', torch.cat((torch.ones(bucket_num - bucket_start, feature_dim),
                                               torch.ones(bucket_num - bucket_start, feature_dim)), dim=1))
        self.register_buffer('num_samples_tracked', torch.zeros(bucket_num - bucket_start))

    @staticmethod
    def _get_kernel_window(kernel, ks, sigma, bins, device):
        assert kernel in ['gaussian', 'triang', 'laplace']
        half_ks = (ks - 1) // 2
        if kernel == 'gaussian':
            base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
            base_kernel = np.array(base_kernel, dtype=np.float32)
            kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / sum(gaussian_filter1d(base_kernel, sigma=sigma))
        elif kernel == 'triang':
            kernel_window = triang(ks) / sum(triang(ks))
        else:
            laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
            kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / sum(map(laplace, np.arange(-half_ks, half_ks + 1)))

        print(f'Using FDS: [{kernel.upper()}] ({ks}/{sigma})')

        window_l, window_r = [], []
        mid_window = [kernel_window[half_ks]] if bins == 1 else [kernel_window[half_ks] / bins] * bins
        for i in range(half_ks):
            tmp_l = kernel_window[i]
            tmp_r = kernel_window[i-half_ks]
            tmp_window_l = [tmp_l] if bins == 1 else [tmp_l / bins] * bins
            tmp_window_r = [tmp_r] if bins == 1 else [tmp_r / bins] * bins
            window_l += tmp_window_l
            window_r += tmp_window_r
        kernel_window_bins = window_l + mid_window + window_r

        print(np.sum(kernel_window))
        print(f'PRM kernel window{kernel_window}')
        print(f'PRM kernel window{kernel_window_bins}')

        kernel_window = kernel_window_bins
        return torch.tensor(kernel_window, dtype=torch.float32).cuda(device=device)

    def _update_last_epoch_stats(self):
        # TODO here
        self.running_mean_last_epoch = self.running_mean
        self.running_var_last_epoch = self.running_var

        mu, var = torch.tensor_split(self.running_mean, 2, dim=1)

        mu_smooth = F.conv1d(
            input=F.pad(mu.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks * self.bins + self.half_bin, self.half_ks * self.bins + self.half_bin),
                        mode='reflect'),
            weight=self.kernel_window.view(1, 1, -1), padding=0
        ).permute(2, 1, 0).squeeze(1)

        var_smooth = F.conv1d(
            input=F.pad(var.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks * self.bins + self.half_bin, self.half_ks * self.bins + self.half_bin),
                        mode='reflect'),
            weight=(self.kernel_window**2).view(1, 1, -1), padding=0
        ).permute(2, 1, 0).squeeze(1)
        
        self.smoothed_mean_last_epoch = torch.cat((mu_smooth, var_smooth), dim=1) 
        
        mu, var = torch.tensor_split(self.running_var, 2, dim=1)
        
        mu_smooth = F.conv1d(
            input=F.pad(mu.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks * self.bins + self.half_bin, self.half_ks * self.bins + self.half_bin),
                        mode='reflect'),
            weight=self.kernel_window.view(1, 1, -1), padding=0
        ).permute(2, 1, 0).squeeze(1)
        
        var_smooth = F.conv1d(
            input=F.pad(var.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks * self.bins + self.half_bin, self.half_ks * self.bins + self.half_bin),
                        mode='reflect'),
            weight=(self.kernel_window**2).view(1, 1, -1), padding=0
        ).permute(2, 1, 0).squeeze(1)
        
        self.smoothed_var_last_epoch = torch.cat((mu_smooth, var_smooth), dim=1)
        
        assert (self.running_var_last_epoch < 0).sum()==0
        assert (self.smoothed_var_last_epoch < 0).sum()==0
        assert (self.running_mean_last_epoch[:,2048:] <= 0).sum()==0
        assert (self.smoothed_mean_last_epoch[:,2048:] <= 0).sum()==0

    # def reset(self):
    #     self.running_mean = torch.cat((torch.zeros(bucket_num - bucket_start, feature_dim),
    #                          torch.ones(bucket_num - bucket_start, feature_dim)), dim=1).to(torch.float32)
    #     self.running_var = torch.cat((torch.ones(bucket_num - bucket_start, feature_dim),
    #                          torch.ones(bucket_num - bucket_start, feature_dim)), dim=1).to(torch.float32)
    #
    #     self.running_mean_last_epoch = torch.cat((torch.zeros(bucket_num - bucket_start, feature_dim),
    #                                 torch.ones(bucket_num - bucket_start, feature_dim)), dim=1).to(torch.float32)
    #     self.running_var_last_epoch = torch.cat((torch.ones(bucket_num - bucket_start, feature_dim),
    #                                 torch.ones(bucket_num - bucket_start, feature_dim)), dim=1).to(torch.float32)
    #
    #     self.smoothed_mean_last_epoch = torch.cat((torch.zeros(bucket_num - bucket_start, feature_dim),
    #                                 torch.ones(bucket_num - bucket_start, feature_dim)), dim=1).to(torch.float32)
    #     self.smoothed_var_last_epoch = torch.cat((torch.ones(bucket_num - bucket_start, feature_dim),
    #                                 torch.ones(bucket_num - bucket_start, feature_dim)), dim=1).to(torch.float32)
    #
    #     self.num_samples_tracked.zero_()

    def update_last_epoch_stats(self, epoch):
        if epoch == self.epoch + 1:
            self.epoch += 1
            self._update_last_epoch_stats()
            print(f"Updated smoothed statistics on Epoch [{epoch}]!")

    def update_running_stats(self, features, labels, epoch):
        if epoch < self.epoch:
            return

        assert self.feature_dim * 2 == features.size(1), "Input feature dimension is not aligned!"
        assert features.size(0) == labels.size(0), "Dimensions of features and labels are not aligned!"

        for label in torch.unique(labels):
            if label > self.bucket_num - 1 or label < self.bucket_start:
                continue
            elif label == self.bucket_start:
                curr_feats = features[labels <= label]
            elif label == self.bucket_num - 1:
                curr_feats = features[labels >= label]
            else:
                curr_feats = features[labels == label]
            
            # TODO here
            curr_num_sample = curr_feats.size(0)
            
            mu, var = torch.tensor_split(curr_feats, 2, dim=1)
            
            
            curr_mean_mu = torch.mean(mu, 0)
            curr_mean_var = torch.sum(var, 0) / curr_num_sample**2
            curr_mean = torch.cat((curr_mean_mu, curr_mean_var))
            
#             curr_var_mu = torch.var(mu, 0, unbiased=True if curr_feats.size(0) != 1 else False)
#             curr_var_var = torch.var(var, 0, unbiased=True if curr_feats.size(0) != 1 else False)
            
            sum_var_mu = var - curr_mean_var + mu**2 - curr_mean_mu**2
            curr_var_mu = torch.sum(sum_var_mu, 0) / curr_num_sample
            
            
#             if curr_num_sample == 1:
#                 print(curr_var_mu.sum())
#                 assert (curr_var_mu<0).sum() == 0.0
    
            if curr_num_sample == 1:
                curr_var_var = torch.zeros_like(curr_mean_mu)
            else:
                curr_var_var = curr_var_mu**2 * 2 / (curr_num_sample-1)

            curr_var = torch.cat((curr_var_mu, curr_var_var))
            
            

            self.num_samples_tracked[int(label - self.bucket_start)] += curr_num_sample
            
            factor = self.momentum if self.momentum is not None else \
                (1 - curr_num_sample / float(self.num_samples_tracked[int(label - self.bucket_start)]))
            
            
            factor = 0 if epoch == self.start_update else factor
            
            self.running_mean[int(label - self.bucket_start)] = \
                (1 - factor) * curr_mean + factor * self.running_mean[int(label - self.bucket_start)]
            self.running_var[int(label - self.bucket_start)] = \
                (1 - factor) * curr_var + factor * self.running_var[int(label - self.bucket_start)]
            
            assert (self.running_var[int(label - self.bucket_start)]<0.0).sum()==0, curr_num_sample

        print(f"Updated running statistics with Epoch [{epoch}] features!")

    def smooth(self, features, labels, epoch):
        if epoch < self.start_smooth:
            return features

        labels = labels.squeeze(1)
        try:
            for label in torch.unique(labels):
                if label > self.bucket_num - 1 or label < self.bucket_start:
                    continue
                elif label == self.bucket_start:
                    features[labels <= label] = calibrate_mean_var_bayes(
                        features[labels <= label],
                        self.running_mean_last_epoch[int(label - self.bucket_start)],
                        self.running_var_last_epoch[int(label - self.bucket_start)],
                        self.smoothed_mean_last_epoch[int(label - self.bucket_start)],
                        self.smoothed_var_last_epoch[int(label - self.bucket_start)])
                elif label == self.bucket_num - 1:
                    features[labels >= label] = calibrate_mean_var_bayes(
                        features[labels >= label],
                        self.running_mean_last_epoch[int(label - self.bucket_start)],
                        self.running_var_last_epoch[int(label - self.bucket_start)],
                        self.smoothed_mean_last_epoch[int(label - self.bucket_start)],
                        self.smoothed_var_last_epoch[int(label - self.bucket_start)])
                else:
                    features[labels == label] = calibrate_mean_var_bayes(
                        features[labels == label],
                        self.running_mean_last_epoch[int(label - self.bucket_start)],
                        self.running_var_last_epoch[int(label - self.bucket_start)],
                        self.smoothed_mean_last_epoch[int(label - self.bucket_start)],
                        self.smoothed_var_last_epoch[int(label - self.bucket_start)])
        except:
            print(label)
            print(self.running_mean_last_epoch.shape)
            print(self.smoothed_mean_last_epoch.shape)
            print(self.running_mean_last_epoch[int(label - self.bucket_start)].shape)
            print(self.smoothed_mean_last_epoch[int(label - self.bucket_start)].shape)
        return features
