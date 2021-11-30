import torch
import cv2
import numpy as np
from datetime import timedelta


def ts2np(x):
    x = x.squeeze(0)
    x = x.permute(1,2,0)
    x = x.cpu().contiguous().numpy()
    return x

def np2ts(x, device="cpu"):
    x = x.transpose(2, 0, 1)
    x = torch.from_numpy(x)
    x = x.unsqueeze(0)
    x = x.to(device)
    return x

def time_elapse(start, end):
    return str(timedelta(seconds = end - start))[:-7]

def blend(f_a, r_bp, alpha=0.8, tau=0.05):
    """
    :param response:
    :param f_a: feature map (either F_A or F_BP)
    :param r_bp: reconstructed feature (R_BP or R_A)
    :param alpha: scalar balance the ratio of content and style in new feature map
    :param tau: threshold, default: 0.05 (suggested in paper)
    :return: (f_a*W + r_bp*(1-W)) where W=alpha*(response>tau)

    Following the official implementation, I replace the sigmoid function (stated in paper) with indicator function
    """
    response = f_a.pow(2).sum(dim=1, keepdim=True)
    response = (response - response.min()) / (response.max() - response.min())
    weight = (response > tau).type(f_a.type()) * alpha
    f_ap = f_a * weight + r_bp * (1. - weight)
    return f_ap


def load_image(file_A, resizeRatio=1.0):
    ori_AL = cv2.imread(file_A)
    ori_img_sizes = ori_AL.shape[:2]

    # resize
    if ori_AL.shape[0] > 700:
        ratio = 700 / ori_AL.shape[0]
        ori_AL = cv2.resize(ori_AL,None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)

    if ori_AL.shape[1] > 700:
        ratio = 700 / ori_AL.shape[1]
        ori_AL = cv2.resize(ori_AL,None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)

    if ori_AL.shape[0] < 200:
        ratio = 700 / ori_AL.shape[0]
        ori_AL = cv2.resize(ori_AL,None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)

    if ori_AL.shape[1] < 200:
        ratio = 700 / ori_AL.shape[1]
        ori_AL = cv2.resize(ori_AL,None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)

    if ori_AL.shape[0]*ori_AL.shape[1] > 350000:
        ratio = np.sqrt(350000 / (ori_AL.shape[1]*ori_AL.shape[0]))
        ori_AL = cv2.resize(ori_AL,None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)

    img_A = cv2.resize(ori_AL,None, fx=resizeRatio, fy=resizeRatio, interpolation=cv2.INTER_CUBIC)

    return img_A