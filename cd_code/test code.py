from codes import cfg
from torch.autograd import Variable
from torchvision.utils import save_image
from matplotlib import pyplot as plt

from data.train_dataset import train_dataset
import utils.loss as ls
from utils import metric as mc
from utils.utils import *

import os
import cv2

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


os.environ["CUDA_VISIBLE_DEVICES"] = '2'
cuda = True if torch.cuda.is_available() else False


def various_distance(out_vec_t0, out_vec_t1, dist_flag):
    if dist_flag == 'l2':
        distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=2)
    if dist_flag == 'l1':
        distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=1)
    if dist_flag == 'cos':
        distance = 1 - F.cosine_similarity(out_vec_t0, out_vec_t1)
    return distance

def single_layer_heatmap_visual(output_t0, output_t1, save_change_map_dir, filename, layer_flag, dist_flag, idx):
    n, c, h, w = output_t0.data.shape
    out_t0_rz = torch.transpose(output_t0.view(c, h * w), 1, 0)
    out_t1_rz = torch.transpose(output_t1.view(c, h * w), 1, 0)
    distance = various_distance(out_t0_rz, out_t1_rz, dist_flag)
    distance = (distance > 1.0).float()
    similar_distance_map = distance.view(h, w).data.cpu().numpy()
    similar_distance_map_rz = Variable(torch.from_numpy(similar_distance_map[np.newaxis, np.newaxis, :]))
    if idx % 10 == 0:
        similar_dis_map_colorize = cv2.applyColorMap(np.uint8(255 * similar_distance_map_rz[0][0].cpu()),
                                                     cv2.COLORMAP_BONE)
        save_change_map_dir_layer = os.path.join(save_change_map_dir, layer_flag)
        check_dir(save_change_map_dir_layer)
        change_map_dir = os.path.join(save_change_map_dir_layer, filename + '.jpg')
        cv2.imwrite(change_map_dir, similar_dis_map_colorize)
    return similar_distance_map_rz.data.cpu().numpy()

def validate(net_CD, val_dataloader, epoch, save_change_map_dir, save_roc_dir):
    net_CD.eval()
    cont_conv_total, total_count = 0.0, 0.0
    metric_for_conditions = init_metric_for_class_for_cmu(1)

    for idx, batch in enumerate(val_dataloader):
        image1, image2, label, name, height, width = batch
        height, width, name = height.numpy()[0], width.numpy()[0], name[0]
        image1, image2, label = Variable(image1.cuda()), Variable(image2.cuda()), Variable(label.cuda())
        image1_norm, image2_norm, label_norm = image1 / 255.0, image2 / 255.0, label / 255.0

        out_conv = net_CD(image1, image2)
        out_t0_conv, out_t1_conv = out_conv

        conv_distance = single_layer_heatmap_visual(out_t0_conv, out_t1_conv, save_change_map_dir, name, key, 'l2', idx)
        cont_conv = mc.RMS_Contrast(conv_distance)

        cont_conv_total += cont_conv

        total_count += 1

        prob_change = conv_distance[0][0]
        gt = label_norm.data.cpu().numpy()
        FN, FP, posNum, negNum = mc.eval_image_rewrite(gt[0], prob_change, cl_index=1)
        metric_for_conditions[0]['total_fp'] += FP
        metric_for_conditions[0]['total_fn'] += FN
        metric_for_conditions[0]['total_posnum'] += posNum
        metric_for_conditions[0]['total_negnum'] += negNum
        cont_conv_mean = cont_conv_total / total_count

    thresh = np.array(range(0,256)) / 255.0
    conds = metric_for_conditions.keys()
    for cond_name in conds:
        total_posnum = metric_for_conditions[cond_name]['total_posnum']
        total_negnum = metric_for_conditions[cond_name]['total_negnum']
        total_fp = metric_for_conditions[cond_name]['total_fp']
        total_fn = metric_for_conditions[cond_name]['total_fn']
        metric_dict = mc.pxEval_maximizeFMeasure(total_posnum,total_negnum, total_fn, total_fp, thresh=thresh)
        metric_for_conditions[cond_name].setdefault('metric', metric_dict)
        metric_for_conditions[cond_name].setdefault('contrast_conv', cont_conv_mean)


    f_score_total = 0.0
    for cond_name in conds:
        pr, rec, f_score = metric_for_conditions[cond_name]['metric']['precision'], metric_for_conditions[cond_name]['metric']['recall'], metric_for_conditions[cond_name]['metric']['MaxF']
        roc_save_epoch_dir = os.path.join(save_roc_dir, str(epoch))
        check_dir(roc_save_epoch_dir)
        mc.save_PTZ_metric2disk(metric_for_conditions[cond_name], roc_save_epoch_dir)
        roc_save_dir = roc_save_epoch_dir + '_' + str(cond_name) + '_roc.png'
        mc.plotPrecisionRecall(pr, rec, roc_save_dir, benchmark_pr=None)
        f_score_total += f_score

    print(f_score_total / (len(conds)))
    return f_score_total/len(conds)


if __name__ == '__main__':

    datas, trainloader, valloader = train_dataset('CDD', crop_size=cfg.INPUT_SIZE, batch_size=cfg.BATCH_SIZE,
                                                  num_workers=cfg.num_workers)

    weights = torch.FloatTensor(datas['weights']).cuda()

    import model.BSFNet as models

    net_CD = models.SiameseNet()
    criterion_CD = ls.BCL_v2()
    criterion_auxillary = ls.BCLwithUncertainty_v1()
    criterion_iou = ls.IOUloss_v2()

    if cuda:
        net_CD.cuda()
        criterion_CD.cuda()
        criterion_auxillary.cuda()
        criterion_iou.cuda()

    learning_rate = 1e-4
    optimizer_CD = torch.optim.Adam(net_CD.parameters(), lr=learning_rate, betas=(cfg.b1, cfg.b2))


    load_checkpoint = torch.load(cfg.SAVE_PATH + '.pth')
    net_CD.load_state_dict(load_checkpoint['net_CD'])
    EPOCH = load_checkpoint['epoch']
    load_best_metric = load_checkpoint['metric']
    print('load_cd_epoch = %d' % (EPOCH))
    start_epoch = EPOCH
    key = 'bsfnet_best'

    save_result_path = os.path.join(cfg.SAVE_PATH, 'test')
    check_dir(save_result_path)
    save_image_path = os.path.join(save_result_path, 'imgs')
    check_dir(save_image_path)
    save_roc_path = os.path.join(save_result_path, 'roc')
    check_dir(save_roc_path)

    current_metric = validate(net_CD, valloader, start_epoch, save_image_path, save_roc_path)

