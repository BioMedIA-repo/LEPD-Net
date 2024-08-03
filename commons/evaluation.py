import matplotlib.pyplot as plt

from commons.utils import *
import numpy as np
from sklearn.utils import compute_class_weight as sk_compute_class_weight
from sklearn.metrics import roc_curve, accuracy_score, recall_score, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, average_precision_score
from scipy.spatial.distance import directed_hausdorff as hausdorff
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
from ignite.metrics import DiceCoefficient, Accuracy, mIoU
import torch
from scipy.spatial import distance
from scipy.stats import ttest_ind
from medpy.metric.binary import hd95
import pickle
from sklearn.metrics import precision_recall_curve


def calculate_metrics(y_test, y_pred):
    """Calculates the accuracy metrics"""

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_pred, y_test)
    recall = recall_score(y_test, y_pred)  # TPR 真阳率
    f1score = f1_score(y_test, y_pred)
    sensitivity = sensitivity_score(y_pred, y_test)
    specificity = specificity_score(y_pred, y_test)  # TNR = 1 - FPR
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    AUC = auc(fpr, tpr)  # x:FPR, y:TPR
    t, P_value = ttest_ind(y_test, y_pred)
    dice = dice_coef(y_test, y_pred)
    js = jaccard(y_test, y_pred)

    return [accuracy, precision, recall, f1score, sensitivity, specificity, AUC, P_value]

def check_metrics(y_test, y_pred):
    same = y_test==y_pred
    different = y_test!=y_pred
    TP = np.sum(same & (y_test==1))
    TN = np.sum(same & (y_test==0))
    FP = np.sum(different & (y_pred==1))
    FN = np.sum(different & (y_pred==0))
    return [TP, TN, FP, FN]

def get_micro_metrics(check, dict=None):
    TP_total = 0
    FP_total = 0
    FN_total = 0
    for key in dict:
        TP_total += check[key][0]
        FP_total += check[key][2]
        FN_total += check[key][-1]
    Precision_micro = TP_total / (TP_total + FP_total)
    Recall_micro = TP_total / (TP_total + FN_total)
    F1_micro = 2 * (Precision_micro * Recall_micro) / (Precision_micro + Recall_micro)
    return [Precision_micro, Recall_micro, F1_micro]

def get_weight_average(eval, dict=None):
    check = eval['check']
    new_eval = []
    for key in dict:
        sample = check[key][0] + check[key][-1]
        total_sample = sum(check[key])
        new_eval.append([num * sample / total_sample for num in eval[key]])
    return np.sum(np.array(new_eval), axis=0)

def calculate_metrics_multi(y_test, y_pred, dict=None):
    eval = {}
    check = {}
    for key in dict:
        # print(row[NAME], row['训练组a验证组b'], row['Fistula'],row['序号'])
        y_dic = np.where(y_test == int(key), 1, 0)
        y_pre = np.where(y_pred == int(key), 1, 0)
        # y_dic = y_test
        # y_pre = y_pred
        eval[key] = calculate_metrics(y_dic, y_pre)
        check[key] = check_metrics(y_dic, y_pre)

    eval['check'] = check
    eval['weight_average'] = get_weight_average(eval, dict)
    eval['micro'] = get_micro_metrics(check, dict)
    return eval


def get_AUC(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)  # x:FPR, y:TPR
    return [roc_auc, fpr, tpr]

def dice_coef(y_true, y_pred):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(y_true).astype(np.bool)
    im2 = np.asarray(y_pred).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


def get_accuracy(SR, GT, threshold=0.5):
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()
    corr = torch.sum(SR == GT)
    tensor_size = torch.prod(torch.tensor(SR.size()))
    acc = float(corr) / float(tensor_size)
    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()

    # TP : True Positive
    # FN : False Negative
    TP = (((SR == 1).int() + (GT == 1).int()).int() == 2).int()
    FN = (((SR == 0).int() + (GT == 1).int()).int() == 2).int()

    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

    return SE


def get_specificity(SR, GT, threshold=0.5):
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()

    # TN : True Negative
    # FP : False Positive
    TN = (((SR == 0).int() + (GT == 0).int()).int() == 2).int()
    FP = (((SR == 1).int() + (GT == 0).int()).int() == 2).int()

    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)

    return SP


def get_precision(SR, GT, threshold=0.5):
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()

    # TP : True Positive
    # FP : False Positive
    TP = (((SR == 1).int() + (GT == 1).int()).int() == 2).int()
    FP = (((SR == 1).int() + (GT == 0).int()).int() == 2).int()

    PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)

    return PC


def get_F1(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    F1 = 2 * SE * PC / (SE + PC + 1e-6)

    return F1


def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()

    Inter = torch.sum((SR + GT) == 2).int()
    Union = torch.sum((SR + GT) >= 1).int()

    JS = float(Inter) / (float(Union) + 1e-6)

    return JS


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()

    Inter = torch.sum((SR + GT) == 2).int()
    DC = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)

    return DC


def correct_predictions(output_probabilities, targets):
    """
    Compute the number of predictions that match some target classes in the
    output of a model.

    Args:
        output_probabilities: A tensor of probabilities for different output
            classes.
        targets: The indices of the actual target classes.

    Returns:
        The number of correct predictions in 'output_probabilities'.
    """
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return out_classes, correct.item()


def jaccard(y_true, y_pred):
    intersect = np.sum(y_true * y_pred)  # Intersection points
    union = np.sum(y_true) + np.sum(y_pred)  # Union points
    return (float(intersect)) / (union - intersect + 1e-7)


def compute_jaccard(y_true, y_pred):
    mean_jaccard = 0.
    thresholded_jaccard = 0.

    for im_index in range(y_pred.shape[0]):
        current_jaccard = jaccard(y_true=y_true[im_index], y_pred=y_pred[im_index])

        mean_jaccard += current_jaccard
        thresholded_jaccard += 0 if current_jaccard < 0.65 else current_jaccard

    mean_jaccard = mean_jaccard / y_pred.shape[0]
    thresholded_jaccard = thresholded_jaccard / y_pred.shape[0]

    return mean_jaccard, thresholded_jaccard


def get_iou(pred, target, threshold=0.5):
    # 将概率图转换为二进制掩模
    pred = (pred > threshold).float()
    target = (target > threshold).float()

    intersection = (pred * target).sum()  # 计算交集
    union = (pred + target).sum() - intersection  # 计算并集
    iou = intersection / union  # 计算IOU

    return iou.item()  # 将Tensor转换为标量


def compute_all_metric_for_seg(y_true, y_pred, thres=0.5):
    # batch, width, height = y_true.size()
    # fpr, tpr, thresholds = roc_curve((y_true), y_pred)
    try:
        accuracy = get_accuracy(y_pred, y_true, threshold=thres)
        dice_score = get_DC(y_pred, y_true, threshold=thres)
        precision = get_precision(y_pred, y_true, threshold=thres)
        sensitivity = get_sensitivity(y_pred, y_true, threshold=thres)
        specificity = get_specificity(y_pred, y_true, threshold=thres)
        F1_score = get_F1(y_pred, y_true, threshold=thres)
        JS_score = get_JS(y_pred, y_true, threshold=thres)
        iou = get_iou(y_pred, y_true, threshold=thres)

        y_pred = y_pred.numpy().flatten()
        y_true = y_true.numpy().flatten()
        y_pred = np.where(y_pred > thres, 1, 0)
        AUC_ROC = roc_auc_score(y_true, y_pred)
        # precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        # precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
        # recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
        # AUC_prec_rec = np.trapz(precision, recall)
        # 算法问题，这里是单张patch，不符合
        # mean_jaccard, thresholded_jaccard = compute_jaccard(np.reshape(y_true, (batch, width, height)),
        #                                                     np.reshape(y_pred, (batch, width, height)))
        # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
        # roc_curve = plt.figure()
        # plt.plot(fpr, tpr, '-', label=algorithm + '_' + dataset + '(AUC = %0.4f)' % AUC_ROC)
        # plt.title('ROC curve', fontsize=14)
        # plt.xlabel("FPR (False Positive Rate)", fontsize=15)
        # plt.ylabel("TPR (True Positive Rate)", fontsize=15)
        # plt.legend(loc="lower right")
        # plt.xticks(fontsize=15)
        # plt.yticks(fontsize=15)
        # Precision-recall curve
        # print("Area under the ROC curve: " + str(AUC_ROC)
        #       + "\nArea under Precision-Recall curve: " + str(AUC_prec_rec)
        #       + "\nMean Jaccard similarity score: " + str(mean_jaccard)
        #       + "\nF1 score (F-measure): " + str(F1_score)
        #       + "\nACCURACY: " + str(accuracy)
        #       + "\nSENSITIVITY: " + str(sensitivity)
        #       + "\nSPECIFICITY: " + str(specificity)
        #       + "\nPRECISION: " + str(precision)
        #       + "\nDICE DIST: " + str(dice_distance)
        #       + "\nDICE SCORE: " + str(dice_score)
        #       )
        return [AUC_ROC, JS_score, F1_score, accuracy,
                sensitivity, specificity, precision, dice_score, iou]
    except ValueError:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]


def accuracy_check(mask, prediction):
    ims = [mask, prediction]
    np_ims = []
    for item in ims:
        if 'PIL' in str(type(item)):
            item = np.array(item)
        elif 'torch' in str(type(item)):
            item = item.cpu().detach().numpy()
        np_ims.append(item)
    compare = np.equal(np.where(np_ims[0] > 0.5, 1, 0), np_ims[1])
    accuracy = np.sum(compare)
    return accuracy / len(np_ims[0].flatten())


def accuracy_pixel_level(output, target):
    """ Computes the accuracy during training and validation for ternary label """
    batch_size = target.shape[0]
    results = np.zeros((7,), np.float)

    for i in range(batch_size):
        pred = output[i, :, :]
        label = target[i, :, :]

        # inside part
        pred_inside = pred == 1
        label_inside = label == 1
        metrics_inside = compute_pixel_level_metrics(pred_inside, label_inside)

        results += np.array(metrics_inside)

    return [value / batch_size for value in results]


def compute_pixel_level_metrics(pred, target):
    """ Compute the pixel-level tp, fp, tn, fn between
    predicted img and groundtruth target
    """

    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(target, np.ndarray):
        target = np.array(target)

    tp = np.sum(pred * target)  # true postives
    tn = np.sum((1 - pred) * (1 - target))  # true negatives
    fp = np.sum(pred * (1 - target))  # false postives
    fn = np.sum((1 - pred) * target)  # false negatives

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    F1 = 2 * precision * recall / (precision + recall + 1e-10)
    acc = (tp + tn) / (tp + fp + tn + fn + 1e-10)
    performance = (recall + tn / (tn + fp + 1e-10)) / 2
    iou = tp / (tp + fp + fn + 1e-10)
    hd_95 = hd95(pred, target) if tp > 0 else 0
    return [acc, hd_95, iou, recall, precision, F1, performance]


def voc_ap(rec, prec, use_07_metric=True):
    '''
    https://github.com/amdegroot/ssd.pytorch/blob/5b0b77faa955c1917b0c710d770739ba8fbff9b7/eval.py#L364
    :param prec: [n]
    :param rec: [n]
    :param use_07_metric: use_o7_metric or use_10_metric
    :return: ap
    '''
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        #
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def iou_xywh_torch(boxes1, boxes2):
    """
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制，且需要是Tensor
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(x, y, w, h)
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    # 分别计算出boxes1和boxes2的左上角坐标、右下角坐标
    # 存储结构为(xmin, ymin, xmax, ymax)，其中(xmin,ymin)是bbox的左上角坐标，(xmax,ymax)是bbox的右下角坐标
    boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
    boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)

    # 计算出boxes1与boxes1相交部分的左上角坐标、右下角坐标
    left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = torch.max(right_down - left_up, torch.zeros_like(right_down))
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU


def nms(bboxes, score_threshold, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes:
    假设有N个bbox的score大于score_threshold，那么bboxes的shape为(N, 6)，存储格式为(xmin, ymin, xmax, ymax, score, class)
    其中(xmin, ymin, xmax, ymax)的大小都是相对于输入原图的，score = conf * prob，class是bbox所属类别的索引号
    :return: best_bboxes
    假设NMS后剩下N个bbox，那么best_bboxes的shape为(N, 6)，存储格式为(xmin, ymin, xmax, ymax, score, class)
    其中(xmin, ymin, xmax, ymax)的大小都是相对于输入原图的，score = conf * prob，class是bbox所属类别的索引号
    """
    classes_in_img = list(set(bboxes[:, 5].astype(np.int32)))  # 把所有bbox的类别取出来，因为 set() 不会重复,有去重功能
    best_bboxes = []

    for cls in classes_in_img:  # 对每个类别进行nms
        cls_mask = (bboxes[:, 5].astype(np.int32) == cls)
        cls_bboxes = bboxes[cls_mask]  # 从所有的预测框中挑出要nms的类别的框
        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])  # 返回 score 最大的bbox的索引
            best_bbox = cls_bboxes[max_ind]  # 挑出最好的框
            best_bboxes.append(best_bbox)  # 把最好的框放进入
            cls_bboxes = np.concatenate(
                [cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])  # 将除了score 最大的bbox之外的其他bboxes取出来
            iou = iou_xyxy_numpy(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])  # 计算score 最大的bbox和其他bbox的iou
            assert method in ['nms', 'soft-nms']
            weight = np.ones((len(iou),), dtype=np.float32)
            if method == 'nms':  # nms方法
                iou_mask = iou > iou_threshold  # 将iou > iou_threshold 的 bbox 置为True
                weight[iou_mask] = 0.0  # 把置为 true 的bbox的索引改为 0
            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))
            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight  # 将置为 true 的bbox的 score 改为 0
            score_mask = cls_bboxes[:, 4] > score_threshold  # 将该类的bbox中score大于score_threshold的bbox提取出来
            cls_bboxes = cls_bboxes[score_mask]  # 得到更新后的bbox
    return np.array(best_bboxes)
