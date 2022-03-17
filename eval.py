import sys

import numpy as np

def TP(h, t):
    return np.sum(np.logical_and(h,t))

def FN(h, t):
    return np.sum(np.logical_and(np.logical_not(h), t))

def FP(h, t):
    return np.sum(h) - TP(h, t)

def TN(h, t):
    return np.sum(np.logical_not(h)) - FN(h,t)

def precision(h, t):
    return TP(h, t) / (TP(h, t) + FP(h, t))

def recall(h, t):
    return TP(h, t) / (TP(h, t) + FN(h, t))

def F1(h, t):
    return 2 * precision(h,t) * recall(h,t) / (precision(h,t) + recall(h,t))

def F_beta(h, t, beta):
    return (1 + beta**2) * precision(h,t) * recall(h,t) / (beta**2 * precision(h, t) + recall(h,t))

def accuracy(h, t):
    # h: hypothesis
    # t: target
    assert(h.shape == t.shape)
    cnt = np.sum(h == t)
    return cnt / len(t)

def full(h,t,beta=1):
    return accuracy(h,t), precision(h, t), recall(h,t), F_beta(h, t, beta)

def output(h, t, beta=1, f=sys.stdout):
    result = full(h,t,beta)
    print("Acc: {:.4}, P: {:.4}, R: {:.4}, F: {:.4} (beta={})".format(*result, beta), file=f)
    return result