import os
import shutil
import torch
import logging
import numpy as np
import random
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
import torch
import torch.nn as nn
import torch.nn.functional as F


def cleanup():
    torch.distributed.destroy_process_group()

def is_main_process():
    return torch.distributed.get_rank() == 0


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def seed_torch(seed=728):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**30 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def query_yes_no(question):
    """ Ask a yes/no question via input() and return their answer. """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    prompt = " [Y/n] "

    while True:
        print(question + prompt, end=':')
        choice = input().lower()
        if choice == '':
            return valid['y']
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def prepare_folders(args):
    # return
    # return
    # dir_path = os.path.join(args.store_root, args.store_name)
    # if not os.path.exists(dir_path):
    #     os.mkdir(dir_path)
    folders_util = [args.store_root, os.path.join(args.store_root, args.store_name)]
    if os.path.exists(folders_util[-1]) and not args.resume and not args.pretrained and not args.evaluate:
        if query_yes_no('overwrite previous folder: {} ?'.format(folders_util[-1])):
            shutil.rmtree(folders_util[-1])
            print(folders_util[-1] + ' removed.')
        else:
            raise RuntimeError('Output folder {} already exists'.format(folders_util[-1]))
    for folder in folders_util:
        if not os.path.exists(folder):
            print(f"===> Creating folder: {folder}")
            os.mkdir(folder)


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(args, state, is_best, prefix=''):
    filename = f"{args.store_root}/{args.store_name}/{prefix}ckpt.pth.tar"
    # torch.save(state, filename)
    if is_best:
        logging.info("===> Saving current best checkpoint...")
        torch.save(state, filename.replace('pth.tar', 'best.pth.tar'))
        # shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))



def calibrate_mean_var(matrix, m1, v1, m2, v2, clip_min=0.1, clip_max=10):
    if torch.sum(v1) < 1e-10:
        return matrix
    if (v1 == 0.).any():
        valid = (v1 != 0.)
        factor = torch.clamp(v2[valid] / v1[valid], clip_min, clip_max)
        matrix[:, valid] = (matrix[:, valid] - m1[valid]) * torch.sqrt(factor) + m2[valid]
        return matrix

    factor = torch.clamp(v2 / v1, clip_min, clip_max)
    return (matrix - m1) * torch.sqrt(factor) + m2


def calibrate_mean_var_bayes(matrix, m1, v1, m2, v2, clip_min=0.1, clip_max=10):
    mu_mu_1, mu_var_1 = torch.tensor_split(m1, 2)
    var_mu_1, _ = torch.tensor_split(v1, 2)
    mu_mu_2, mu_var_2 = torch.tensor_split(m2, 2)
    var_mu_2, _ = torch.tensor_split(v2, 2)
    
    if torch.sum(var_mu_1) < 1e-10:
        return matrix
    
    matrix_mu, matrix_var = torch.tensor_split(matrix, 2, dim=1)

    
    if (var_mu_1 == 0.).any():
        valid = (var_mu_1 != 0.)
        factor = torch.clamp(var_mu_2[valid] / var_mu_1[valid], clip_min, clip_max)
        matrix_mu = (matrix_mu[:, valid] - mu_mu_1[valid]) * torch.sqrt(factor) + mu_mu_2[valid]
        matrix_var = (matrix_var[:, valid] + mu_var_1[valid]) * torch.sqrt(factor) + mu_var_2[valid]
#         assert (matrix_var<=0).sum() == 0.0, valid
        matrix = torch.cat((matrix_mu, matrix_var), dim=1)
        return matrix
    
    
    factor = torch.clamp(var_mu_2 / var_mu_1, clip_min, clip_max)
    
    matrix_mu = (matrix_mu - mu_mu_1) * torch.sqrt(factor) + mu_mu_2
    matrix_var = (matrix_var + mu_var_1) * torch.sqrt(factor) + mu_var_2

    matrix = torch.cat((matrix_mu, matrix_var), dim=1)
    return matrix


def get_lds_kernel_window(kernel, ks, sigma, bins):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    window_l, window_r = [], []
    mid_window = [kernel_window[half_ks]] if bins == 1 else [kernel_window[half_ks] / bins] * bins
    for i in range(half_ks):
        tmp_l = kernel_window[i]
        tmp_r = kernel_window[i - half_ks]
        tmp_window_l = [tmp_l] if bins == 1 else [tmp_l / bins] * bins
        tmp_window_r = [tmp_r] if bins == 1 else [tmp_r / bins] * bins
        window_l += tmp_window_l
        window_r += tmp_window_r
    kernel_window_bins = window_l + mid_window + window_r

    print(f'LDS kernel window{kernel_window}')
    print(f'LDS kernel window{kernel_window_bins}')

    kernel_window = kernel_window_bins
    return kernel_window
