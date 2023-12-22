import os, sys, logging
import time, glob, argparse, random
import pandas as pd  # must import
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
import torchvision
import torch.optim as optim
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from resnet import resnet50, Calibration_model
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import json
from zipfile import ZipFile
import numpy as np
import scipy
from collections import defaultdict
from scipy.stats import spearmanr, gmean
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from sklearn.metrics import mean_squared_error, roc_auc_score
from copy import deepcopy

from tensorboard_logger import Logger
from loss import *
from datasets import AgeDB
from utils import *
from evidential_deep_learning import *



# os.environ["KMP_WARNINGS"] = "FALSE"
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("--local_rank", default=os.environ['LOCAL_RANK'], type=int)
# imbalanced related
# LDS
parser.add_argument('--lds', action='store_true', default=True, help='whether to enable LDS')
parser.add_argument('--lds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='LDS kernel type')
parser.add_argument('--lds_ks', type=int, default=5, help='LDS kernel size: should be odd number')
parser.add_argument('--lds_sigma', type=float, default=2, help='LDS gaussian/laplace kernel sigma')
# FDS
parser.add_argument('--fds', action='store_true', default=True, help='whether to enable FDS')
parser.add_argument('--fds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='FDS kernel type')
parser.add_argument('--fds_ks', type=int, default=5, help='FDS kernel size: should be odd number')
parser.add_argument('--fds_sigma', type=float, default=2, help='FDS gaussian/laplace kernel sigma')
parser.add_argument('--start_update', type=int, default=0, help='which epoch to start FDS updating')
parser.add_argument('--start_smooth', type=int, default=1, help='which epoch to start using FDS to smooth features')
parser.add_argument('--bucket_num', type=int, default=100, help='maximum bucket considered for FDS')
parser.add_argument('--bucket_start', type=int, default=3, choices=[0, 3],
                    help='minimum(starting) bucket for FDS, 0 for IMDBWIKI, 3 for AgeDB')
parser.add_argument('--fds_mmt', type=float, default=0.9, help='FDS momentum')

# re-weighting: SQRT_INV / INV
parser.add_argument('--reweight', type=str, default='sqrt_inv', choices=['none', 'sqrt_inv', 'inverse'],
                    help='cost-sensitive reweighting scheme')
# two-stage training: RRT
parser.add_argument('--retrain_fc', action='store_true', default=False,
                    help='whether to retrain last regression layer (regressor)')

# use prm & edl & recons
parser.add_argument('--use_prm', action='store_true', default=True, help='whether to use prm')
parser.add_argument('--use_edl', action='store_true', default=True, help='whether to use edl')
parser.add_argument('--use_cdm', action='store_true', default=True, help='whether to use cdm')
parser.add_argument('--use_recons', action='store_true', default=True, help='whether to use reconstructor')
parser.add_argument('--lambda_recons', type=float, default=0.7, help='lambda for recons_loss')
parser.add_argument('--lambda_reg', type=float, default=0.01, help='lambda for reg_loss')

# training/optimization related
parser.add_argument('--dataset', type=str, default='agedb', choices=['imdb_wiki', 'agedb'], help='dataset name')
parser.add_argument('--data_dir', type=str, default='./data/', help='data directory')
parser.add_argument('--model', type=str, default='resnet50', help='model name')
parser.add_argument('--store_root', type=str, default='/data/local/ziyan/checkpoint', help='root path for storing checkpoints, logs')
parser.add_argument('--store_name', type=str, default='', help='experiment store name')
parser.add_argument('--gpu', type=int, default=6)
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='optimizer type')
parser.add_argument('--loss', type=str, default='l1', choices=['mse', 'l1', 'focal_l1', 'focal_mse', 'huber'],
                    help='training loss type')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--epoch', type=int, default=90, help='number of epochs to train')
parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='optimizer weight decay')
parser.add_argument('--schedule', type=int, nargs='*', default=[60, 80], help='lr schedule (when to drop lr by 10x)')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--print_freq', type=int, default=10, help='logging frequency')
parser.add_argument('--img_size', type=int, default=224, help='image size used in training')
parser.add_argument('--workers', type=int, default=32, help='number of workers used in data loading')
parser.add_argument('--seeds', type=int, default=1223, help='randomness seed')  # 1223
# checkpoints
parser.add_argument('--resume', type=str, default='', help='checkpoint file path to resume training')
parser.add_argument('--pretrained', type=str, default='', help='checkpoint file path to load backbone weights')
parser.add_argument('--evaluate', action='store_true', help='evaluate only flag')

parser.set_defaults(augment=True)
args, unknown = parser.parse_known_args()


args.start_epoch, args.best_loss = 0, 1e5

if len(args.store_name):
    args.store_name = f'_{args.store_name}'
if not args.lds and args.reweight != 'none':
    args.store_name += f'_{args.reweight}'
if args.lds:
    args.store_name += f'_lds_{args.lds_kernel[:3]}_{args.lds_ks}'
    if args.lds_kernel in ['gaussian', 'laplace']:
        args.store_name += f'_{args.lds_sigma}'
if args.fds:
    if args.use_prm:
        args.store_name += f'_prm_{args.fds_kernel[:3]}_{args.fds_ks}'
    else:
        args.store_name += f'_fds_{args.fds_kernel[:3]}_{args.fds_ks}'
    if args.fds_kernel in ['gaussian', 'laplace']:
        args.store_name += f'_{args.fds_sigma}'
    args.store_name += f'_{args.start_update}_{args.start_smooth}_{args.fds_mmt}'
if args.retrain_fc:
    args.store_name += f'_retrain_fc'
args.store_name = f"{args.lambda_reg}_{args.seeds}_{args.lambda_recons}_{args.dataset}_{args.model}{args.store_name}_{args.optimizer}_{args.loss}_{args.lr}_{args.batch_size}"

if args.use_edl:
    if args.use_cdm:
        args.store_name += f"_cdm"
    else:
        args.store_name += f"_edl"

if args.use_recons:
    args.store_name += f"_recons"

print(args.store_name)

prepare_folders(args)
#
logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.store_root, args.store_name, 'training.log')),
        logging.StreamHandler()
    ])
print = logging.info
print(f"Args: {args}")
print(f"Store name: {args.store_name}")

tb_logger = Logger(logdir=os.path.join(args.store_root, args.store_name), flush_secs=2)


prior_dict = {'gamma': 0, 'v': 0, 'alpha': 0.3, 'beta': 0}

torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(precision=10)

device = torch.device(f'cuda:{args.gpu}')

def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.4f')
    if args.use_edl:
        losses = AverageMeter(f'Loss (EDL)', ':.3f')
    else:
        losses = AverageMeter(f'Loss ({args.loss.upper()})', ':.3f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch)
    )

    KL_losses = []
    l1_losses = []
    count_losses = []
    criterion_l1 = nn.L1Loss()

    model.train()
    set_requires_grad(model, True)
    end = time.time()
    # train_loader.sampler.set_epoch(epoch)
    for idx, (inputs, targets, weights) in enumerate(train_loader):
        data_time.update(time.time() - end)
        inputs, targets, weights = \
            inputs.cuda(device=device, non_blocking=True), targets.cuda(device=device, non_blocking=True), weights.cuda(device=device, non_blocking=True)

        if args.fds:
            if args.use_recons:
                outputs, mu_z, log_var_z, _, x_recons = model(inputs, targets, epoch)
            else:
                outputs, mu_z, log_var_z, _ = model(inputs, targets, epoch)
        else:
            if args.use_recons:
                outputs, mu_z, log_var_z, x_recons = model(inputs, targets, epoch)
            else:
                outputs, mu_z, log_var_z = model(inputs, targets, epoch)

        if args.use_edl:
            if args.use_cdm:
                loss = globals()["cdm_loss"](outputs, targets, args.lambda_reg, True, weights, **prior_dict)
                gamma, _, _, _ = globals()["get_normal_output"](outputs, True, weights, **prior_dict)
            else:
                loss = globals()["weighted_edl_loss"](outputs, targets, args.lambda_reg, weights)
                gamma, _, _, _ = torch.tensor_split(outputs, 4, dim=1)
            l1_losses.append(criterion_l1(gamma, targets) * inputs.size(0))
            count_losses.append(inputs.size(0))
        else:
            loss = globals()[f"weighted_{args.loss}_loss"](outputs, targets, weights)

        # if args.use_prm:
        #     kl_loss = globals()["KL_DIV"](mu_z, log_var_z, 0.001)
        #     loss += kl_loss
        #     KL_losses.append(kl_loss)

        if args.use_recons:
            loss += F.mse_loss(inputs, x_recons) * args.lambda_recons

        assert not (np.isnan(loss.item()) or loss.item() > 1e6), f"Loss explosion: {loss.item()}"

        losses.update(loss.item(), inputs.size(0))

        # if idx % 2:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if idx % args.print_freq == 0:
            progress.display(idx)

    if args.fds and epoch >= args.start_update:
        print(f"Create Epoch [{epoch}] features of all training data...")
        encodings, labels = [], []
        with torch.no_grad():
            for (inputs, targets, _) in train_loader:
                inputs = inputs.cuda(device=device, non_blocking=True)
                if args.use_recons:
                    outputs, _, _, feature, _ = model(inputs, targets, epoch)
                else:
                    outputs, _, _, feature = model(inputs, targets, epoch)
                encodings.extend(feature.data.squeeze().cpu().numpy())
                labels.extend(targets.data.squeeze().cpu().numpy())

        encodings, labels = torch.from_numpy(np.vstack(encodings)).cuda(device=device), \
                            torch.from_numpy(np.hstack(labels)).cuda(device=device)
        model.module.FDS.update_last_epoch_stats(epoch)
        model.module.FDS.update_running_stats(encodings, labels, epoch)

    # print("kl:")
    # print(torch.mean(torch.tensor(KL_losses)))
    # print("l1: ")
    # print(torch.tensor(l1_losses).sum() / torch.tensor(count_losses).sum())
    return losses.avg


def eval_uncertainty(y_true, y_pred, y_var, metric='l2'):
    if metric == 'l1':
        func = lambda a, b: np.abs(a - b)
    else:
        func = lambda a, b: np.sqrt((a - b) ** 2)

    error = func(y_pred, y_true)

    idx_var = torch.argsort(torch.FloatTensor(y_var), descending=True).numpy()
    idx_error = torch.argsort(torch.FloatTensor(error), descending=True).numpy()

    y_var_cp = np.hstack(y_var)
    error_cp = np.hstack(error)

    var_idx_list = []
    error_idx_list = []

    y = []
    x = []

    for i in range(y_var_cp.shape[0] - 1):
        removed_var_idx = idx_var[i]
        removed_error_idx = idx_error[i]

        var_idx_list.append(removed_var_idx)
        error_idx_list.append(removed_error_idx)

        y.append(np.mean(np.delete(error_cp, var_idx_list)))
        x.append(np.mean(np.delete(error_cp, error_idx_list)))

    z = np.hstack(y) - np.hstack(x)
    z = np.abs(z)
    z = z / np.max(z)
    idx = np.arange(len(y))
    idx = idx / idx[-1]
    ause = np.trapz(z, x=idx)

    nll = 0.5 * (np.log(y_var) + (y_true-y_pred)**2 / y_var) + np.log(np.sqrt(2*np.pi))

    return nll.mean(), ause


def validate(val_loader, model, cali_model=None, train_labels=None, prefix='Val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses_mse = AverageMeter('Loss (MSE)', ':.3f')
    losses_l1 = AverageMeter('Loss (L1)', ':.3f')

    if args.use_edl:
        losses_edl = AverageMeter('Loss (EDL)', ':.3f')
        losses_NIG_NLL = AverageMeter('Loss (NIG_NLL)', ':.3f')
        losses_NIG_Reg = AverageMeter('Loss (NIG_Reg)', ':.3f')

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses_l1, losses_edl, losses_NIG_NLL, losses_NIG_Reg],
            prefix=f'{prefix}: '
        )
    else:
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses_mse, losses_l1],
            prefix=f'{prefix}: '
        )

    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_gmean = nn.L1Loss(reduction='none')

    model.eval()
    set_requires_grad(model, False)
    if cali_model is not None:
        set_requires_grad(cali_model, False)
    losses_all = []
    preds, labels = [], []
    variance = []
    edl_params = {'vs': [], 'alphas': [], 'betas': []}
    with torch.no_grad():
        end = time.time()
        for idx, (inputs, targets, _) in enumerate(val_loader):
            inputs, targets = inputs.cuda(device=device, non_blocking=True), targets.cuda(device=device, non_blocking=True)
            labels.extend(targets.data.cpu().numpy())
            if args.use_edl:
                outputs = model(inputs)
                if args.use_cdm:
                    gamma, v, alpha, beta = globals()["get_normal_output"](outputs, False, None, **prior_dict)
                else:
                    gamma, v, alpha, beta = torch.tensor_split(outputs, 4, dim=1)

                outputs_normal = torch.cat((gamma, v, alpha, beta), dim=1)

                if prefix == 'Test':
                    if cali_model is not None:
                        outputs_normal = cali_model(outputs_normal)
                    gamma, v, alpha, beta = torch.tensor_split(outputs_normal, 4, dim=1)

                preds.extend(gamma.data.cpu().numpy())
                if prefix == 'Test':
                    var = beta / ((alpha - 1) * v)
                    variance.extend(var.data.cpu().numpy())
                edl_params['vs'].extend(v.data.cpu().numpy())
                edl_params['alphas'].extend(alpha.data.cpu().numpy())
                edl_params['betas'].extend(beta.data.cpu().numpy())
                loss_mse = criterion_mse(gamma, targets)
                loss_l1 = criterion_l1(gamma, targets)
                loss_all = criterion_gmean(gamma, targets)
                loss_edl_combo = globals()["EvidentialRegression"](outputs_normal, targets, args.lambda_reg)
                losses_edl.update(loss_edl_combo[0].item(), inputs.size(0))
                losses_NIG_NLL.update(loss_edl_combo[1].item(), inputs.size(0))
                losses_NIG_Reg.update(loss_edl_combo[2].item(), inputs.size(0))


            else:
                outputs = model(inputs)
                preds.extend(outputs.data.cpu().numpy())
                loss_mse = criterion_mse(outputs, targets)
                loss_l1 = criterion_l1(outputs, targets)
                loss_all = criterion_gmean(outputs, targets)

            losses_all.extend(loss_all.cpu().numpy())
            losses_mse.update(loss_mse.item(), inputs.size(0))
            losses_l1.update(loss_l1.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if idx % args.print_freq == 0:
                progress.display(idx)

        loss_gmean = gmean(np.hstack(losses_all), axis=None).astype(float)

        labels, preds = np.hstack(labels), np.hstack(preds)

        if args.use_edl:
            for key, val in edl_params.items():
                edl_params[key] = np.hstack(val).reshape(-1, 1)

            if prefix == 'Test':
                variance = np.hstack(variance)
                nllgau, ause = eval_uncertainty(labels, preds, variance)

                shot_dict = shot_metrics(preds, labels, train_labels, variance=variance, edl_params=edl_params)
                print(
                    f" * Overall: MSE {losses_mse.avg:.3f}\tL1 {losses_l1.avg:.3f}\tG-Mean {loss_gmean:.3f}\tEDL {losses_edl.avg:.3f}\t"
                    f"NIG_NLL {losses_NIG_NLL.avg:.3f}\tNIG_Reg {losses_NIG_Reg.avg:.3f}\t"
                    f"nllgau {nllgau:.3f}\tAUSE {ause:.3f}")
                print(f" * Many: MSE {shot_dict['many']['mse']:.3f}\tL1 {shot_dict['many']['l1']:.3f}\t"
                      f"G-Mean {shot_dict['many']['gmean']:.3f}\tEDL {shot_dict['many']['edl']:.3f}\t"
                      f"NIG_NLL {shot_dict['many']['NLL']:.3f}\tNIG_Reg {shot_dict['many']['Reg']:.3f}\t"
                      f"nllgau {shot_dict['many']['nllgau']:.3f}\tAUSE {shot_dict['many']['ause']:.3f}")
                print(f" * Median: MSE {shot_dict['median']['mse']:.3f}\tL1 {shot_dict['median']['l1']:.3f}\t"
                      f"G-Mean {shot_dict['median']['gmean']:.3f}\tEDL {shot_dict['median']['edl']:.3f}\t"
                      f"NIG_NLL {shot_dict['median']['NLL']:.3f}\tNIG_Reg {shot_dict['median']['Reg']:.3f}\t"
                      f"nllgau {shot_dict['median']['nllgau']:.3f}\tAUSE {shot_dict['median']['ause']:.3f}")
                print(f" * Low: MSE {shot_dict['low']['mse']:.3f}\tL1 {shot_dict['low']['l1']:.3f}\t"
                      f"G-Mean {shot_dict['low']['gmean']:.3f}\tEDL {shot_dict['low']['edl']:.3f}\t"
                      f"NIG_NLL {shot_dict['low']['NLL']:.3f}\tNIG_Reg {shot_dict['low']['Reg']:.3f}\t"
                      f"nllgau {shot_dict['low']['nllgau']:.3f}\tAUSE {shot_dict['low']['ause']:.3f}")

            else:
                shot_dict = shot_metrics(np.hstack(preds), np.hstack(labels), train_labels, edl_params=edl_params)

                print(
                    f" * Overall: MSE {losses_mse.avg:.3f}\tL1 {losses_l1.avg:.3f}\tG-Mean {loss_gmean:.3f}\tEDL {losses_edl.avg:.3f}\t"
                    f"NIG_NLL {losses_NIG_NLL.avg:.3f}\tNIG_Reg {losses_NIG_Reg.avg:.3f}")
                print(f" * Many: MSE {shot_dict['many']['mse']:.3f}\tL1 {shot_dict['many']['l1']:.3f}\t"
                      f"G-Mean {shot_dict['many']['gmean']:.3f}\tEDL {shot_dict['many']['edl']:.3f}\t"
                      f"NIG_NLL {shot_dict['many']['NLL']:.3f}\tNIG_Reg {shot_dict['many']['Reg']:.3f}")
                print(f" * Median: MSE {shot_dict['median']['mse']:.3f}\tL1 {shot_dict['median']['l1']:.3f}\t"
                      f"G-Mean {shot_dict['median']['gmean']:.3f}\tEDL {shot_dict['median']['edl']:.3f}\t"
                      f"NIG_NLL {shot_dict['median']['NLL']:.3f}\tNIG_Reg {shot_dict['median']['Reg']:.3f}")
                print(f" * Low: MSE {shot_dict['low']['mse']:.3f}\tL1 {shot_dict['low']['l1']:.3f}\t"
                      f"G-Mean {shot_dict['low']['gmean']:.3f}\tEDL {shot_dict['low']['edl']:.3f}\t"
                      f"NIG_NLL {shot_dict['low']['NLL']:.3f}\tNIG_Reg {shot_dict['low']['Reg']:.3f}")
        else:
            shot_dict = shot_metrics(np.hstack(preds), np.hstack(labels), train_labels)
            print(f" * Overall: MSE {losses_mse.avg:.3f}\tL1 {losses_l1.avg:.3f}\tG-Mean {loss_gmean:.3f}")
            print(f" * Many: MSE {shot_dict['many']['mse']:.3f}\t"
                  f"L1 {shot_dict['many']['l1']:.3f}\tG-Mean {shot_dict['many']['gmean']:.3f}")
            print(f" * Median: MSE {shot_dict['median']['mse']:.3f}\t"
                  f"L1 {shot_dict['median']['l1']:.3f}\tG-Mean {shot_dict['median']['gmean']:.3f}")
            print(f" * Low: MSE {shot_dict['low']['mse']:.3f}\t"
                  f"L1 {shot_dict['low']['l1']:.3f}\tG-Mean {shot_dict['low']['gmean']:.3f}")

    if args.use_edl:
        #         return losses_mse.avg, losses_l1.avg, loss_gmean
        return losses_mse.avg, losses_l1.avg, loss_gmean, losses_edl.avg, losses_NIG_NLL.avg, losses_NIG_Reg.avg
    else:
        return losses_mse.avg, losses_l1.avg, loss_gmean


def calibrate_val(val_loader, model):
    cali_model = Calibration_model()
    cali_model = nn.DataParallel(cali_model, device_ids=[args.gpu])

    cali_optimizer = torch.optim.Adam(cali_model.parameters(), lr=0.001)

    model.eval()
    set_requires_grad(model, False)
    set_requires_grad(cali_model, True)

    for idx, (inputs, targets, _) in enumerate(val_loader):
        inputs, targets = inputs.cuda(device=device, non_blocking=True), targets.cuda(device=device,
                                                                                      non_blocking=True)
        outputs = model(inputs)
        if args.use_cdm:
            gamma, v, alpha, beta = globals()["get_normal_output"](outputs, False, None, **prior_dict)
            print(alpha.mean())

        else:
            gamma, v, alpha, beta = torch.tensor_split(outputs, 4, dim=1)

        cali_input = torch.cat([gamma, v, alpha, beta], dim=1)
        cali_out = cali_model(cali_input)

        gamma, v, alpha, beta = torch.tensor_split(cali_out, 4, dim=1)

        nll_loss = NIG_NLL(targets, gamma, v, alpha, beta).mean()

        cali_optimizer.zero_grad()
        nll_loss.backward()
        cali_optimizer.step()

        t1, t2, t3 = cali_model.module.get_weights()
        print(f"the cali weights are: {t1}, {t2}, {t3}")

    return cali_model


def shot_metrics(preds, labels, train_labels, variance=None, edl_params=None, many_shot_thr=100, low_shot_thr=20):
    train_labels = np.array(train_labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError(f'Type ({type(preds)}) of predictions not supported')

    if variance is not None:
        many_shot_labels, median_shot_labels, low_shot_labels = [], [], []
        many_shot_preds, median_shot_preds, low_shot_preds = [], [], []
        many_shot_vars, median_shot_vars, low_shot_vars = [], [], []

    train_class_count, test_class_count = [], []
    mse_per_class, l1_per_class, l1_all_per_class = [], [], []
    if args.use_edl:
        edl_per_class = []
        NIG_NLL_per_class = []
        NIG_Reg_per_class = []
        glob_output = np.hstack((preds.reshape(-1, 1), edl_params['vs'], edl_params['alphas'], edl_params['betas']))
        glob_output = torch.tensor(glob_output).to(device)

    for l in np.unique(labels):
        train_class_count.append(len(train_labels[train_labels == l]))
        test_class_count.append(len(labels[labels == l]))

        if args.use_edl:
            mask = np.array([labels == l]).reshape(-1, )
            l_output = glob_output[mask, :]
            l_labels = torch.tensor(labels[labels == l]).to(device).view(-1, 1)
            loss_edl_combo = globals()["EvidentialRegression"](l_output, l_labels.reshape(-1, 1), args.lambda_reg)
            edl_per_class.append(loss_edl_combo[0].item())
            NIG_NLL_per_class.append(loss_edl_combo[1].item())
            NIG_Reg_per_class.append(loss_edl_combo[2].item())

        if variance is not None:
            if train_class_count[-1] > many_shot_thr:
                many_shot_labels.append(labels[labels == l])
                many_shot_preds.append(preds[labels == l])
                many_shot_vars.append(variance[labels == l])
            elif train_class_count[-1] < low_shot_thr:
                low_shot_labels.append(labels[labels == l])
                low_shot_preds.append(preds[labels == l])
                low_shot_vars.append(variance[labels == l])
            else:
                median_shot_labels.append(labels[labels == l])
                median_shot_preds.append(preds[labels == l])
                median_shot_vars.append(variance[labels == l])

        mse_per_class.append(np.sum((preds[labels == l] - labels[labels == l]) ** 2))
        l1_per_class.append(np.sum(np.abs(preds[labels == l] - labels[labels == l])))
        l1_all_per_class.append(np.abs(preds[labels == l] - labels[labels == l]))

    many_shot_mse, median_shot_mse, low_shot_mse = [], [], []
    many_shot_l1, median_shot_l1, low_shot_l1 = [], [], []
    many_shot_gmean, median_shot_gmean, low_shot_gmean = [], [], []
    many_shot_cnt, median_shot_cnt, low_shot_cnt = [], [], []

    if args.use_edl:
        many_shot_edl, median_shot_edl, low_shot_edl = [], [], []
        many_shot_NLL, median_shot_NLL, low_shot_NLL = [], [], []
        many_shot_Reg, median_shot_Reg, low_shot_Reg = [], [], []

    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot_mse.append(mse_per_class[i])
            many_shot_l1.append(l1_per_class[i])
            many_shot_gmean += list(l1_all_per_class[i])
            many_shot_cnt.append(test_class_count[i])

            if args.use_edl:
                many_shot_edl.append(edl_per_class[i] * test_class_count[i])
                many_shot_NLL.append(NIG_NLL_per_class[i] * test_class_count[i])
                many_shot_Reg.append(NIG_Reg_per_class[i] * test_class_count[i])

        elif train_class_count[i] < low_shot_thr:
            low_shot_mse.append(mse_per_class[i])
            low_shot_l1.append(l1_per_class[i])
            low_shot_gmean += list(l1_all_per_class[i])
            low_shot_cnt.append(test_class_count[i])

            if args.use_edl:
                low_shot_edl.append(edl_per_class[i] * test_class_count[i])
                low_shot_NLL.append(NIG_NLL_per_class[i] * test_class_count[i])
                low_shot_Reg.append(NIG_Reg_per_class[i] * test_class_count[i])
        else:
            median_shot_mse.append(mse_per_class[i])
            median_shot_l1.append(l1_per_class[i])
            median_shot_gmean += list(l1_all_per_class[i])
            median_shot_cnt.append(test_class_count[i])

            if args.use_edl:
                median_shot_edl.append(edl_per_class[i] * test_class_count[i])
                median_shot_NLL.append(NIG_NLL_per_class[i] * test_class_count[i])
                median_shot_Reg.append(NIG_Reg_per_class[i] * test_class_count[i])

    shot_dict = defaultdict(dict)
    shot_dict['many']['mse'] = np.sum(many_shot_mse) / np.sum(many_shot_cnt)
    shot_dict['many']['l1'] = np.sum(many_shot_l1) / np.sum(many_shot_cnt)
    shot_dict['many']['gmean'] = gmean(np.hstack(many_shot_gmean), axis=None).astype(float)
    shot_dict['median']['mse'] = np.sum(median_shot_mse) / np.sum(median_shot_cnt)
    shot_dict['median']['l1'] = np.sum(median_shot_l1) / np.sum(median_shot_cnt)
    shot_dict['median']['gmean'] = gmean(np.hstack(median_shot_gmean), axis=None).astype(float)
    shot_dict['low']['mse'] = np.sum(low_shot_mse) / np.sum(low_shot_cnt)
    shot_dict['low']['l1'] = np.sum(low_shot_l1) / np.sum(low_shot_cnt)
    shot_dict['low']['gmean'] = gmean(np.hstack(low_shot_gmean), axis=None).astype(float)

    if args.use_edl:
        shot_dict['many']['edl'] = np.sum(many_shot_edl) / np.sum(many_shot_cnt)
        shot_dict['median']['edl'] = np.sum(median_shot_edl) / np.sum(median_shot_cnt)
        shot_dict['low']['edl'] = np.sum(low_shot_edl) / np.sum(low_shot_cnt)
        shot_dict['many']['NLL'] = np.sum(many_shot_NLL) / np.sum(many_shot_cnt)
        shot_dict['median']['NLL'] = np.sum(median_shot_NLL) / np.sum(median_shot_cnt)
        shot_dict['low']['NLL'] = np.sum(low_shot_NLL) / np.sum(low_shot_cnt)
        shot_dict['many']['Reg'] = np.sum(many_shot_Reg) / np.sum(many_shot_cnt)
        shot_dict['median']['Reg'] = np.sum(median_shot_Reg) / np.sum(median_shot_cnt)
        shot_dict['low']['Reg'] = np.sum(low_shot_Reg) / np.sum(low_shot_cnt)

    if variance is not None:
        many_nllgau, many_ause = eval_uncertainty(np.hstack(many_shot_labels), np.hstack(many_shot_preds),
                                                np.hstack(many_shot_vars))
        median_nllgau, median_ause = eval_uncertainty(np.hstack(median_shot_labels), np.hstack(median_shot_preds),
                                                    np.hstack(median_shot_vars))
        low_nllgau, low_ause = eval_uncertainty(np.hstack(low_shot_labels), np.hstack(low_shot_preds),
                                              np.hstack(low_shot_vars))
        shot_dict['many']['nllgau'] = many_nllgau
        shot_dict['many']['ause'] = many_ause
        shot_dict['median']['nllgau'] = median_nllgau
        shot_dict['median']['ause'] = median_ause
        shot_dict['low']['nllgau'] = low_nllgau
        shot_dict['low']['ause'] = low_ause

    return shot_dict


def main():
    # Data
    print('=====> Preparing data...')
    print(f"File (.csv): {args.dataset}.csv")
    df = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}.csv"))
    df_train, df_val, df_test = df[df['split'] == 'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    train_labels = df_train['age']

    g = torch.Generator()
    g.manual_seed(args.seeds)

    train_dataset = AgeDB(data_dir=args.data_dir, df=df_train, img_size=args.img_size, split='train',
                          reweight=args.reweight, lds=args.lds, lds_kernel=args.lds_kernel, lds_ks=args.lds_ks,
                          lds_sigma=args.lds_sigma)
    val_dataset = AgeDB(data_dir=args.data_dir, df=df_val, img_size=args.img_size, split='val')
    test_dataset = AgeDB(data_dir=args.data_dir, df=df_test, img_size=args.img_size, split='test')

    # torch.distributed.init_process_group(backend="gloo", world_size=4)
    # datasampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=4, rank=args.local_rank, drop_last=False)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, worker_init_fn=worker_init_fn,
                              num_workers=args.workers, generator=g, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, drop_last=False)
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")

    # Model
    print('=====> Building model...')
    model = resnet50(fds=args.fds, bucket_num=args.bucket_num, bucket_start=args.bucket_start,
                     start_update=args.start_update, start_smooth=args.start_smooth, kernel=args.fds_kernel,
                     ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt, use_edl=args.use_edl,
                     use_cdm=args.use_cdm, use_prm=args.use_prm, use_recons=args.use_recons, device=device).to(device)

    # model = DistributedDataParallel(model, find_unused_parameters=True, device_ids=[args.local_rank], output_device=args.local_rank)

    model = nn.DataParallel(model, device_ids=[args.gpu])

    # evaluate only
    if args.evaluate:
        assert args.resume, 'Specify a trained model using [args.resume]'
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"===> Checkpoint '{args.resume}' loaded (epoch [{checkpoint['epoch']}]), testing...")
        validate(test_loader, model, train_labels=train_labels, prefix='Test')
        return

    if args.retrain_fc:
        assert args.reweight != 'none' and args.pretrained
        print('===> Retrain last regression layer only!')
        for name, param in model.named_parameters():
            if 'fc' not in name and 'linear' not in name:
                param.requires_grad = False

    # Loss and optimizer
    if not args.retrain_fc:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == 'adam' else \
            torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        # optimize only the last linear layer
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        names = list(filter(lambda k: k is not None,
                            [k if v.requires_grad else None for k, v in model.module.named_parameters()]))
        assert 1 <= len(parameters) <= 2  # fc.weight, fc.bias
        print(f'===> Only optimize parameters: {names}')
        optimizer = torch.optim.Adam(parameters, lr=args.lr) if args.optimizer == 'adam' else \
            torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if 'linear' not in k and 'fc' not in k:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        print(f'===> Pretrained weights found in total: [{len(list(new_state_dict.keys()))}]')
        print(f'===> Pre-trained model loaded: {args.pretrained}')

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"===> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume) if args.gpu is None else \
                torch.load(args.resume, map_location=torch.device(f'cuda:{str(args.gpu)}'))
            args.start_epoch = checkpoint['epoch']
            args.best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"===> Loaded checkpoint '{args.resume}' (Epoch [{checkpoint['epoch']}])")
        else:
            print(f"===> No checkpoint found at '{args.resume}'")

    cudnn.benchmark = True


    for epoch in range(args.start_epoch, args.epoch):
        adjust_learning_rate(optimizer, epoch, args)
        for param_group in optimizer.param_groups:
            print("this_lr: ")
            print(param_group['lr'])
        train_loss = train(train_loader, model, optimizer, epoch)

        if args.use_edl:
            val_loss_mse, val_loss_l1, val_loss_gmean, val_loss_edl, val_loss_NLL, val_loss_Reg = validate(val_loader,
                                                                                                           model,
                                                                                                           train_labels=train_labels)
        else:
            val_loss_mse, val_loss_l1, val_loss_gmean = validate(val_loader, model, train_labels=train_labels)

        if args.use_edl:
            # loss_metric = val_loss_NLL + 0.5 * val_loss_l1
            # loss_metric = val_loss_edl
            loss_metric = val_loss_mse if args.loss == 'mse' else val_loss_l1
        else:
            loss_metric = val_loss_mse if args.loss == 'mse' else val_loss_l1
        is_best = loss_metric < args.best_loss
        args.best_loss = min(loss_metric, args.best_loss)
        if args.use_edl:
            #             print(f"Best {'L1' if 'l1' in args.loss else 'MSE'} Loss: {args.best_loss:.3f}")
            print(f"Best {'EDL'} Loss: {args.best_loss:.3f}")
        else:
            print(f"Best {'L1' if 'l1' in args.loss else 'MSE'} Loss: {args.best_loss:.3f}")

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'model': args.model,
            'best_loss': args.best_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best)

        if args.use_edl:
            print(f"Epoch #{epoch}: Train loss [{train_loss:.4f}]; "
                  f"Val loss: MSE [{val_loss_mse:.4f}], L1 [{val_loss_l1:.4f}], G-Mean [{val_loss_gmean:.4f}], EDL [{val_loss_edl:.4f}], "
                  f"NIG_NLL [{val_loss_NLL:.3f}], NIG_Reg [{val_loss_Reg:.3f}]")
        else:
            print(f"Epoch #{epoch}: Train loss [{train_loss:.4f}]; "
                  f"Val loss: MSE [{val_loss_mse:.4f}], L1 [{val_loss_l1:.4f}], G-Mean [{val_loss_gmean:.4f}]")

        tb_logger.log_value('train_loss', train_loss, epoch)
        tb_logger.log_value('val_loss_mse', val_loss_mse, epoch)
        tb_logger.log_value('val_loss_l1', val_loss_l1, epoch)
        tb_logger.log_value('val_loss_gmean', val_loss_gmean, epoch)
        if args.use_edl:
            tb_logger.log_value('val_loss_edl', val_loss_edl, epoch)

    print("=" * 120)
    print("Test best model on testset...")
    checkpoint = torch.load(f"{args.store_root}/{args.store_name}/ckpt.best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded best model, epoch {checkpoint['epoch']}, best val loss {checkpoint['best_loss']:.4f}")
    if args.use_edl:
        # if args.use_cdm and args.use_prm:
        #     cali_model = calibrate_val(val_loader, model)
        # else:
        #     cali_model = None
        cali_model = None
        test_loss_mse, test_loss_l1, test_loss_gmean, test_loss_edl, test_loss_NLL, test_loss_Reg = validate(
            test_loader, model, cali_model, train_labels=train_labels, prefix='Test')
        print(
            f"Test loss: MSE [{test_loss_mse:.4f}], L1 [{test_loss_l1:.4f}], G-Mean [{test_loss_gmean:.4f}], EDL [{test_loss_edl:.4f}], "
            f"NIG_NLL [{test_loss_NLL:.3f}], NIG_Reg [{test_loss_Reg:.3f}]\nDone")
    else:
        test_loss_mse, test_loss_l1, test_loss_gmean = validate(test_loader, model, train_labels=train_labels,
                                                                prefix='Test')
        print(f"Test loss: MSE [{test_loss_mse:.4f}], L1 [{test_loss_l1:.4f}], G-Mean [{test_loss_gmean:.4f}]\nDone")

    # cleanup()

main()
