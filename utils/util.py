from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment as linear_assignment
# from sklearn.utils.linear_assignment_ import linear_assignment # LOOK I DON'T HAVE THIS VERSION
import scipy.io
from tqdm import tqdm
import random
import os
import argparse


import torch
import numpy as np


def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor


class SinkhornKnopp(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_iters = args.num_iters_sk
        self.epsilon = args.epsilon_sk

    @torch.no_grad()
    def iterate(self, Q):
        Q = shoot_infs(Q)
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / Q.shape[1]  # Samples

        # obtain permutation/order from the marginals
        marginals_argsort = torch.argsort(Q.sum(1))
        marginals_argsort = marginals_argsort.detach()
        r = []
        for i in range(Q.shape[0]):  # Classes
            r.append((1 / 1) ** (i / (Q.shape[0] - 1.0)))
        r = np.array(r)
        r = r * (Q.shape[1] / Q.shape[0])  # Per-class distribution in the mini-batch
        r = torch.from_numpy(r).cuda(non_blocking=True)
        r[marginals_argsort] = torch.sort(r)[0]  # Sort/permute based on the data order
        r = torch.clamp(r, min=1)  # Clamp the min to have a balance distribution for the tail classes
        r /= r.sum()  # Scaling to make it prob

        for it in range(self.num_iters):
            u = torch.sum(Q, dim=1)
            u = r / u
            u = shoot_infs(u)
            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    @torch.no_grad()
    def forward(self, logits):
        # get assignments
        # import pdb; pdb.set_trace()
        q = logits / self.epsilon
        M = torch.max(q)
        q -= M
        q = torch.exp(q).t()
        return self.iterate(q)

#######################################################
# Evaluate Critiron
#######################################################
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def cluster_acc(y_true, y_pred, return_ind=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind_arr, jnd_arr = linear_assignment(w.max() - w)
    ind = np.array(list(zip(ind_arr, jnd_arr)))

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind

    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class BCE(nn.Module):
    eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.
    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))
        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()

def PairEnum(x,mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
    if mask is not None:
        xmask = mask.view(-1, 1).repeat(1, x.size(1))
        #dim 0: #sample, dim 1:#feature
        x1 = x1[xmask].view(-1, x.size(1))
        x2 = x2[xmask].view(-1, x.size(1))
    return x1, x2

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class CentroidTracker(object):
    def __init__(self, model, labeled_train_loader, num_labeled_classes, device,
                 dataset_name, train_stage, save_root, mode='dynamic'):
        self.model = model
        self.loader = labeled_train_loader
        self.num_labeled_classes = num_labeled_classes
        self.device = device
        self.dataset_name = dataset_name
        self.train_stage = train_stage
        self.root = save_root+'/'
        self.stats_dir = os.path.join(self.root, (self.dataset_name+'_'+self.train_stage+'_stats'))
        if not os.path.exists(self.stats_dir):
            os.makedirs(self.stats_dir)
        self.file_prefix = self.stats_dir+'/'+'epoch'
        self.mode = mode
        self.flying_mean = None
        self.flying_sig = None
        self.flying_cov = None

    def initialize_stats(self, init_mean, init_sig, init_cov):
        self.flying_mean = init_mean
        self.flying_sig = init_sig
        self.flying_cov = init_cov
        print("...... Initialized centroids from the old feature space")

    def generate(self, current_epoch, save_featmap=True):
        # create individual containers for extracted feature, labels, class mean, class sig and class cov
        all_feat = []
        all_labels = []
        class_mean = torch.zeros(self.num_labeled_classes, 512)
        class_sig = torch.zeros(self.num_labeled_classes, 512)

        # extract the feat and label using the current learned model
        self.model.eval()
        print("Start to calculate the statistics of the labeled features for epoch: [{}]".format(current_epoch))
        print("Extract labeled features")
        for batch_idx, (x, label, idx) in enumerate(tqdm(self.loader)):
            # print("---extracting from batch [{}]".format(batch_idx))
            x, label = x.to(self.device), label.to(self.device)
            _, _, feat = self.model(x)
            all_feat.append(feat.detach().clone())
            all_labels.append(label.detach().clone())

        # organize it a bit
        # print(len(all_feat))
        # print(all_feat[0].shape)
        all_feat = torch.cat(all_feat, dim=0).cpu()
        all_labels = torch.cat(all_labels, dim=0).cpu()

        # check the shapes
        print("all_feats shape: {}".format(all_feat.shape))
        print("all_labels shape: {}".format(all_labels.shape))

        print("Calculate labeled Mean-Var-Cov")
        for i in range(self.num_labeled_classes):
            this_feat = all_feat[all_labels==i]
            this_mean = this_feat.mean(dim=0)
            this_var = this_feat.var(dim=0)
            class_mean[i,:] = this_mean
            class_sig[i,:] = (this_var + 1e-5).sqrt()

        ### Calculate Class-Wise Cov
        N = all_feat.size(0)
        C = self.num_labeled_classes
        A = all_feat.size(1)

        NxCxFeatures = all_feat.view(N, 1, A).expand(N, C, A)
        onehot = torch.zeros(N, C)  # .cuda()
        onehot.scatter_(1, all_labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        class_cov = torch.bmm(var_temp.permute(1, 2, 0),
                              var_temp.permute(1, 0, 2)).div(Amount_CxA.view(C, A, 1).expand(C, A, A))

        if self.mode == 'dynamic' and current_epoch != 0:
            self.flying_mean = class_mean.cuda()
            self.flying_sig = class_sig.cuda()
            self.flying_cov = class_cov.cuda()

        this_epoch = {
                        'class_mean': class_mean.cpu().numpy(),
                        'class_sig': class_sig.cpu().numpy(),
                        'class_cov': class_cov.cpu().numpy(),
                        'all_feats': all_feat.numpy(),
                        'all_labels': all_labels.numpy()
                       }
        if save_featmap:
            scipy.io.savemat(self.file_prefix+'{}.mat'.format(current_epoch), this_epoch)
        print("Class mean, sig, cov, feats, labels saved at epoch [{}]".format(current_epoch))
        # return this_epoch
        # comment above to save memory
        return True

    def sample_labeled_features(self, dataset_name):
        if self.mode == 'static':
            print("This centroid tracker in in static mode. It can't sample feantures.")
            return False
        if self.flying_mean is None or self.flying_sig is None or self.flying_cov is None:
            print("Centroid tracker does not have correct statistics, pls check")
            return False

        feats = []
        labels = []

        if dataset_name == 'cifar10':
            num_per_class = 20
        elif dataset_name == 'cifar100':
            num_per_class = 2
        else:
            num_per_class = 3

        for i in range(self.num_labeled_classes):
            dist = torch.distributions.Normal(self.flying_mean[i], self.flying_sig.mean(dim=0))
            this_feat = dist.sample((num_per_class,)).cuda()  # new API
            this_label = torch.ones(this_feat.size(0)).cuda() * i

            feats.append(this_feat)
            labels.append(this_label)

        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0).long()
        # print("..... sampled dynamic centroids in the current feature space.")

        return feats.cuda(), labels.cuda()

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]