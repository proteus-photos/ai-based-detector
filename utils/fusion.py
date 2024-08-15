# ------------------------------------------------------------------------------
# Reference: https://github.com/grip-unina/ClipBased-SyntheticImageDetection/blob/main/utils/fusion.py
# ------------------------------------------------------------------------------
import numpy as np
from scipy.special import logsumexp, logit, log_expit
softplusinv = lambda x: np.log(np.expm1(x))  # log(exp(x)-1)
softminusinv = lambda x: x - np.log(-np.expm1(x)) # info: https://jiafulow.github.io/blog/2019/07/11/softplus-and-softminus/

fusion_functions = {
    'mean_logit'   : lambda x, axis: np.mean(x, axis),
    'max_logit'    : lambda x, axis: np.max(x, axis),
    'median_logit' : lambda x, axis: np.median(x, axis),
    'lse_logit'    : lambda x, axis: logsumexp(x, axis),
    'mean_prob'    : lambda x, axis: softminusinv(logsumexp(log_expit(x), axis) - np.log(x.shape[axis])),
    'soft_or_prob' : lambda x, axis: -softminusinv(np.sum(log_expit(-x), axis)),
}

def apply_fusion(x, typ, axis):
    return fusion_functions[typ](x, axis)
