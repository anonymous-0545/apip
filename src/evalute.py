import numpy as np
import copy
from math import sqrt

import torch
from scipy import stats
from sklearn import preprocessing,metrics




def rmse(y,f):
    """
    Task:    To compute root mean squared error (RMSE)

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  rmse   RSME
    """

    rmse = sqrt(((y - f)**2).mean(axis=0))

    return rmse

def rmse_gpu(y,f):
    """
    Task:    To compute root mean squared error (RMSE)

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  rmse   RSME
    """

    rmse = torch.sqrt(((y - f)**2).mean(axis=0)).cpu().item()

    return rmse


def pearsonr(x, y):
    """
    Mimics `scipy.stats.pearsonr`

    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor

    Returns
    -------
    r_val : float
        pearsonr correlation coefficient between x and y

    Scipy docs ref:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html

    Scipy code ref:
        https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
    Example:
import eir.metrics        >>> x = np.random.randn(100)
        >>> y = np.random.randn(100)
        >>> sp_corr = eir.metrics.pearsonr(x, y)[0]
        >>> th_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
        >>> np.allclose(sp_corr, th_corr)
    """
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val


def pearson(y,f):
    """
    Task:    To compute Pearson correlation coefficient

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  rp     Pearson correlation coefficient
    """

    rp = np.corrcoef(y, f)[0,1]

    return rp

def pearson_gpu(y,f):
    """
    Task:    To compute Pearson correlation coefficient

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  rp     Pearson correlation coefficient
    """

    rp = pearsonr(y, f).cpu().item()

    return rp


def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    tmp = x.argsort().cuda()
    ranks = torch.zeros_like(tmp, device='cuda')

    ranks[tmp] = torch.arange(len(x), device='cuda')
    return ranks


def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
    """Compute correlation between 2 1-D vectors
    Args:
        x: Shape (N, )
        y: Shape (N, )
    """
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)

    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n ** 2 - 1.0)
    del x_rank, y_rank
    return 1.0 - (upper / down)

def spearman_gpu(y,f):
    """
    Task:    To compute Spearman's rank correlation coefficient

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  rs     Spearman's rank correlation coefficient
    """

    rs = spearman_correlation(y, f).cpu().item()

    return rs


def spearman(y,f):
    """
    Task:    To compute Spearman's rank correlation coefficient

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  rs     Spearman's rank correlation coefficient
    """

    rs = stats.spearmanr(y, f)[0]

    return rs




def ci(y,f):
    """
    Task:    To compute concordance index (CI)

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  ci     CI

    References:
    [1] Tapio Pahikkala, Antti Airola, Sami Pietila, Sushil Shakyawar,
    Agnieszka Szwajda, JingTang and Tero Aittokallio.
    Toward more realistic drug-target interaction predictions.
    Briefings in Bioinformatics, 16, pages 325-337, 2014.
    """

    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]

    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0

    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1

    ci = S/z

    return ci




def f1(y,f):
    """
    Task:    To compute F1 score using the threshold of 7 M
             to binarize pKd's into true class labels.

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  f1     F1 score
    """

    y_binary = copy.deepcopy(y)
    y_binary = preprocessing.binarize(y_binary.values.reshape(1,-1), threshold=7.0, copy=False)[0]
    f_binary = copy.deepcopy(f)
    f_binary = preprocessing.binarize(f_binary.values.reshape(1,-1), threshold=7.0, copy=False)[0]

    f1 = metrics.f1_score(y_binary, f_binary)

    return f1




def average_AUC(y,f):

    """
    Task:    To compute average area under the ROC curves (AUC) given ten
             interaction threshold values from the pKd interval [6 M, 8 M]
             to binarize pKd's into true class labels.

    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])

    Output:  avAUC   average AUC

    """

    thr = np.linspace(6,8,10)
    auc = np.empty(np.shape(thr)); auc[:] = np.nan

    for i in range(len(thr)):
        y_binary = copy.deepcopy(y)
        y_binary = preprocessing.binarize(y_binary.values.reshape(1,-1), threshold=thr[i], copy=False)[0]
        fpr, tpr, thresholds = metrics.roc_curve(y_binary, f, pos_label=1)
        auc[i] = metrics.auc(fpr, tpr)

    avAUC = np.mean(auc)

    return avAUC