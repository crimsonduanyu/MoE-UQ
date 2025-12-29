from __future__ import division
from torch import nn
import torch
from sklearn.metrics import r2_score
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
import torch.nn.functional as F



def pinball_loss(y, y_hat, alpha, HuberDelta=0.2, reduction='sum', zero_inflated=True):
    """
    HuberDelta=0.2, zero_inflated=True in Exp
    HuberDelta=None,  zero_inflated=False in LossAblation then it's vanilla Quantile Loss function
    as well as called quantile loss, giving the quantile loss for a single quantile
    """
    support = torch.ones(y.shape).to(y.device)
    if zero_inflated:
        zero_idx = y == 0
        zero_ratio = torch.sum(zero_idx) / y.nelement() + 1e-4
        nonzero_ratio = 1 - zero_ratio + 2e-4
        support[zero_idx] = zero_ratio
        support[~zero_idx] = nonzero_ratio


    if HuberDelta is None:
        loss_fn = torch.nn.L1Loss(reduction='none')
    else:
        loss_fn = torch.nn.HuberLoss(reduction='none', delta=HuberDelta)
    pred_upper = y_hat >= y
    pred_lower = ~pred_upper
    if reduction == 'none':
        pbl = torch.sum(alpha[pred_lower] * loss_fn(y[pred_lower], y_hat[pred_lower])/support[pred_lower]) + \
              torch.sum((1 - alpha)[pred_upper] * loss_fn(y[pred_upper], y_hat[pred_upper])/support[pred_upper])
    else:
        pbl = torch.sum(alpha * loss_fn(y[pred_lower], y_hat[pred_lower])/support[pred_lower]) +\
               torch.sum((1 - alpha) * loss_fn(y[pred_upper], y_hat[pred_upper]/support[pred_upper]))
    return pbl



def JointPinballLoss(y, y_hat_list, alpha_list, HuberDelta=None, use_continuous=False):
    """
    Giving the JointPinballLoss for a list of quantiles
    :param y: y_true
    :param y_hat_list: a list cosisting of y_hat under different quantiles corresponding to alpha_list
    :param alpha_list:a list of quantiles
    :return: JointPinballLoss
    """

    loss = None
    for index, alpha in enumerate(alpha_list):
        y_hat = y_hat_list[:, :,index, ...]
        if loss is None:
            loss = pinball_loss(y, y_hat, alpha)
        else:
            loss += pinball_loss(y, y_hat, alpha)

    return loss



def F1_score_zeropart(y_true, y_pred, threshold=0.5):
    # 计算零值部分的F1分数
    from sklearn.metrics import f1_score
    true_zeros = y_true == 0
    pred_zeros = y_pred < threshold

    return f1_score(true_zeros, pred_zeros, average='weighted')


def R_squared(y_true, y_pred):
    # 计算R方
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    return r2_score(y_true, y_pred, force_finite=True)


def get_accuracy(y_true, y_pred):
    y_true = np.round(y_true.ravel())
    y_pred = np.round(y_pred.ravel())
    return np.mean(np.equal(y_true, y_pred))


def MAE(y_true, y_pred):
    y_pred[y_pred < 0.5] = 0
    return np.mean(np.abs(y_true - y_pred))

mae_loss = nn.L1Loss(reduction='mean')



def RMSE(y_true, y_pred):
    y_pred[y_pred < 0.5] = 0
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


def calculate_cwc(y_true, lower, upper, mu=0.90, eta=1):
    """
    计算Coverage Width Criterion (CWC)
    
    参数:
    y_true (array-like): 真实值数组
    lower (array-like): 预测区间下界数组
    upper (array-like): 预测区间上界数组
    mu (float): 目标覆盖率，默认0.95
    eta (float): 惩罚系数，默认1
    
    返回:
    float: CWC值
    """
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    
    # 检查输入长度一致
    assert len(y_true) == len(lower) == len(upper), "输入长度不一致"
        
    # 计算PICP
    covered1 = np.logical_and(y_true >= lower, y_true <= upper)
    covered2 = np.logical_and(y_true <= lower, y_true >= upper)
    covered = np.logical_or(covered1, covered2)
    picp = np.mean(covered)
    
    # 计算平均宽度和PINAW
    widths = np.abs(upper - lower)
    mean_pi_width = np.mean(widths)
    y_range = np.max(y_true) - np.min(y_true)
    if y_range == 0:
        pinaw = mean_pi_width  # 避免除以0
    else:
        pinaw = mean_pi_width / y_range
    
    # 确定gamma
    gamma = 0 if picp >= mu else 1
    
    # 计算指数项和CWC
    exp_term = np.exp(-eta * (picp - mu))
    cwc = pinaw * (1 + gamma * exp_term)
    
    return cwc

def MPIW_and_PICP(y_true, y_pred_lower, y_pred_upper, mu=0.9, eta=10):
    # 计算MPIW和PICP
    
    y_true = np.array(y_true).ravel()
    y_pred_upper = np.array(y_pred_upper).ravel()
    y_pred_lower = np.array(y_pred_lower).ravel()


    # 打印每个百分位点的计算结果
    hits = ((np.greater_equal(y_true, y_pred_lower) & np.less_equal(y_true, y_pred_upper)) |
            (np.greater_equal(y_pred_lower, y_true) & np.less_equal(y_pred_upper, y_true)))
    hits = hits.astype(np.float32)
    PICP = np.mean(hits)
    MPIW = np.mean(np.abs(y_pred_upper-y_pred_lower))
    CWC = calculate_cwc(y_true, y_pred_lower, y_pred_upper, mu=mu, eta=eta)
    return MPIW, PICP, CWC


