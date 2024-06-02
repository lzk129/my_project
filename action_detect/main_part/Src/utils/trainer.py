import torch
from torch.autograd import Variable
from datetime import datetime
import os
import cv2
import numpy as np
import time
import scipy
import scipy.ndimage
from tqdm import tqdm

# from apex import amp
import torch.nn.functional as F


def eval_mae(y_pred, y):
    """
    evaluate MAE (for test or validation phase)
    :param y_pred:
    :param y:
    :return: Mean Absolute Error
    """
    return torch.abs(y_pred - y).mean()


def F_measur(dataset):
    res_path = 'Result/2017-FPN-New/{}/'.format(dataset)
    gt_path = 'F:/dataset/TPR_NET/TestDataset/{}/GT/'.format(dataset)
    res_list = os.listdir(res_path)
    P = [0 for _ in range(len(res_list))]
    R = [0 for _ in range(len(res_list))]

    for i in range(len(res_list)):
        r_name = res_path + res_list[i]
        g_name = gt_path + res_list[i]
        res = cv2.imread(r_name)
        res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
        h, w = res.shape

        gt = cv2.imread(g_name)
        gt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
        _, gt = cv2.threshold(gt, 125, 255, cv2.THRESH_BINARY)
        th = 2 * sum(sum(res)) / (h * w)  ##阈值
        _, res_tmp = cv2.threshold(res, th, 255, cv2.THRESH_BINARY)
        tmp = res_tmp - gt
        FP = sum(sum(tmp == 255))
        FN = sum(sum(tmp == -255))
        TP = sum(sum((res_tmp == gt) & (gt == 255)))
        P[i] = TP / (TP + FP)
        R[i] = TP / (TP + FN)

    belt2 = 0.3
    p = sum(P) / len(P)
    r = sum(R) / len(R)
    Fmeasure = ((1 + belt2) * p * r) / (belt2 * p + r)
    return Fmeasure

def weighted_F_measur(D , G):
    E = np.abs(G - D)
    gt_mask = np.isclose(G , 1)
    not_gt_mask = np.logical_not(gt_mask)
    dist , idx = scipy.ndimage.distance_transform_edt(not_gt_mask , return_indices = True)

    #Pixel dependency
    Et = np.array(E)
    Et[not_gt_mask] = E[idx[0 , not_gt_mask] , idx[1 , not_gt_mask]]
    sigma = 5.0
    EA = scipy.ndimage.gaussian_filter(Et, sigma=sigma, truncate= 3/ sigma, mode='constant', cval=0.0)

    min_EA = np.minimum(E , EA , where=gt_mask, out=np.array(E))

    #Pixel importance
    B = np.ones(G.shape)
    B[not_gt_mask] = 2 - np.exp(np.log(1 - 0.5) / 5 * dist[not_gt_mask])
    Ew = min_EA * B


    #Final metric computation
    eps = np.spacing(1)
    TPw = np.sum(G) - np.sum(Ew[gt_mask])
    FPw = np.sum(Ew[not_gt_mask])
    Recall_w = 1 - np.mean(Ew[gt_mask])
    Precision_w = TPw / (eps + TPw + FPw)
    # print(Ew)
    # print(Recall_w)
    # print(Precision_w)
    beta2 = 0.3
    F_bw = (1 + beta2) * (Precision_w * Recall_w) / (beta2 * Precision_w + Recall_w + eps)
    return F_bw

def S_Measure(D, G):
    def _object(pred, gt):
        temp = pred[gt == 1]
        x = temp.mean()
        sigma_x = temp.std()
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)
        if torch.isnan(score):
            raise
        return score

    def _centroid(gt):
        rows, cols = gt.size()[-2:]
        gt = gt.view(rows, cols)
        total = gt.sum()
        i = torch.from_numpy(np.arange(0,cols)).float().cuda()
        j = torch.from_numpy(np.arange(0,rows)).float().cuda()
        X = torch.round((gt.sum(dim=0)*i).sum() / total)
        Y = torch.round((gt.sum(dim=1)*j).sum() / total)

        return X.long(), Y.long()

    def _divideGT(gt, X, Y):
        h, w = gt.size()[-2:]
        area = h*w
        gt = gt.view(h, w)
        LT = gt[:Y, :X]
        RT = gt[:Y, X:w]
        LB = gt[Y:h, :X]
        RB = gt[Y:h, X:w]
        X = X.float()
        Y = Y.float()
        w1 = X * Y / area
        w2 = (w - X) * Y / area
        w3 = X * (h - Y) / area
        w4 = 1 - w1 - w2 - w3

        return LT, RT, LB, RB, w1, w2, w3, w4

    def _dividePrediction(pred, X, Y):
        h, w = pred.size()[-2:]
        pred = pred.view(h, w)
        LT = pred[:Y, :X]
        RT = pred[:Y, X:w]
        LB = pred[Y:h, :X]
        RB = pred[Y:h, X:w]

        return LT, RT, LB, RB

    def _S_object(pred, gt):
        fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
        bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
        o_fg = _object(fg, gt)
        o_bg = _object(bg, 1 - gt)
        u = gt.mean()
        Q = u * o_fg + (1 - u) * o_bg
        return Q

    def _S_region(pred, gt):
        X, Y = _centroid(gt)
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = _divideGT(gt, X, Y)
        p1, p2, p3, p4 = _dividePrediction(pred, X, Y)
        Q1 = _ssim(p1, gt1)
        Q2 = _ssim(p2, gt2)
        Q3 = _ssim(p3, gt3)
        Q4 = _ssim(p4, gt4)
        Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
        return Q

    def _ssim(pred, gt):
        gt = gt.float()
        h, w = pred.size()[-2:]
        N = h * w
        x = pred.mean()
        y = gt.mean()
        sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
        sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-20)
        sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)

        aplha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

        if aplha != 0:
            Q = aplha / (beta + 1e-20)
        elif aplha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0

        return Q

    alpha = 0.7
    result = alpha * _S_object(D , G) + (1 - alpha) * _S_region(D , G)
    return result

def E_measure(D , G):
    def _eval_e(y_pred, y, num):
        score = torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_pred_th = (y_pred >= thlist[i]).float()
            if torch.mean(y) == 0.0:  # the ground-truth is totally black
                y_pred_th = torch.mul(y_pred_th, -1)
                enhanced = torch.add(y_pred_th, 1)
            elif torch.mean(y) == 1.0:  # the ground-truth is totally white
                enhanced = y_pred_th
            else:  # normal cases
                fm = y_pred_th - y_pred_th.mean()
                gt = y - y.mean()
                align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
                enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4

            score[i] = torch.sum(enhanced) / (y.numel() - 1 + 1e-20)
        return score

    def _eval_adp_e(y_pred, y):
        th = y_pred.mean() * 2
        y_pred_th = (y_pred >= th).float()
        if torch.mean(y) == 0.0:  # the ground-truth is totally black
            y_pred_th = torch.mul(y_pred_th, -1)
            enhanced = torch.add(y_pred_th, 1)
        elif torch.mean(y) == 1.0:  # the ground-truth is totally white
            enhanced = y_pred_th
        else:  # normal cases
            fm = y_pred_th - y_pred_th.mean()
            gt = y - y.mean()
            align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
            enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
        return torch.sum(enhanced) / (y.numel() - 1 + 1e-20)

    avg_e = 0.0
    adp_e = 0.0
    Q = _eval_e(D , G , 255)
    adp_e = _eval_adp_e(D , G)
    return Q.mean() , adp_e





def numpy2tensor(numpy):
    """
    convert numpy_array in cpu to tensor in gpu
    :param numpy:
    :return: torch.from_numpy(numpy).cuda()
    """
    return torch.from_numpy(numpy).cuda()


def clip_gradient(optimizer, grad_clip):
    """
    recalibrate the misdirection in the training
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay




def trainer(train_loader, model, optimizer, epoch, opt, loss_func, total_step):
    """
    Training iteration
    :param train_loader:
    :param model:
    :param optimizer:
    :param epoch:
    :param opt:
    :param loss_func:
    :param total_step:
    :return:
    """
    model.train()
    for step, data_pack in enumerate(train_loader):
        optimizer.zero_grad()
        images, gts = data_pack
        images = Variable(images).cuda()
        # print(images.shape)
        gts = Variable(gts).cuda()

        cam_s_g, cam_im, cam_sm, _ = model(images)
        loss_sm = loss_func(cam_sm, gts)
        loss_im = loss_func(cam_im, gts)
        loss_s_g = loss_func(cam_s_g, gts)
        loss_total = loss_sm + loss_im + loss_s_g

        # with amp.scale_loss(loss_total, optimizer) as scale_loss:
        #     scale_loss.backward()
        loss_total.backward()
        # clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if step % 10 == 0 or step == total_step:
            print('[{}] => [Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] => [Loss_s: {:.4f} Loss_i: {:0.4f} loss_s_g: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, step, total_step, loss_sm.data, loss_im.data, loss_s_g.data))

    save_path = opt.save_model
    os.makedirs(save_path, exist_ok=True)

    if (epoch+1) % opt.save_epoch == 0:
        torch.save(model.state_dict(), save_path + 'SINet_new_%d.pth' % (epoch+1))
