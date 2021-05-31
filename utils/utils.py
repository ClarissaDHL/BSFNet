import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import json
import cv2
import os

__all__ = ['check_dir', 'adjust_learning_rate', 'save2json', 'load_metric_json',
           'init_metric_for_class_for_cmu', 'rz_label', 'start_record', 'netParams']

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def adjust_learning_rate(learning_rate, optimizer, step, init_lr = 1e-5):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if step <= 150000 and learning_rate>=1e-6:
        lr = init_lr - (2e-6 * (step // 75000))
    elif step > 150000 and learning_rate>1e-6:
        lr = 0.7 * init_lr - (1e-6 * (step // 75000))
    else:
        lr = 1e-6
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save2json(metric_dict,save_path):
    file_ = open(save_path,'w')
    file_.write(json.dumps(metric_dict,ensure_ascii=False,indent=2))
    file_.close()

def load_metric_json(json_path):
    with open(json_path,'r') as f:
        metric = json.load(f)
    return  metric

def init_metric_for_class_for_cmu(number_class):

    metric_for_class = {}
    for i in range(number_class):
        metric_for_each = {}
        thresh = np.array(range(0, 256)) / 255.0
        total_fp = np.zeros(thresh.shape)
        total_fn = np.zeros(thresh.shape)
        metric_for_each.setdefault('total_fp', total_fp)
        metric_for_each.setdefault('total_fn', total_fn)
        metric_for_each.setdefault('total_posnum', 0)
        metric_for_each.setdefault('total_negnum', 0)
        metric_for_class.setdefault(i, metric_for_each)
    return metric_for_class

def rz_label(label, size):

    gt_e = torch.unsqueeze(label,dim=1)
    interp = F.upsample(gt_e,(size[0],size[1]),mode='bilinear')
    gt_rz = torch.squeeze(interp,dim=1)
    return gt_rz

def start_record(model, key = '', log_path=''):

    scoreFileLoc = os.path.join(log_path, key + 'score.txt')
    sampleFileLoc = os.path.join(log_path, key + 'sample.txt')

    if os.path.isfile(scoreFileLoc):
        logger1 = open(scoreFileLoc, 'a')
    else:
        logger1 = open(scoreFileLoc, 'w')
        logger1.write('Network: %s' % (key))
        logger1.write('\nParameters: %d K' %(netParams(model)))
        logger1.write('\n%s\t\t%s\t%s\t\t%s\t%s'
                     % ('Epoch', 'Precision', 'Recall', 'Accuracy', 'Fscores'))

    if os.path.isfile(sampleFileLoc):
        logger2 = open(sampleFileLoc, 'a')
    else:
        logger2 = open(sampleFileLoc, 'w')
        logger2.write('Network: %s' % (key))
        logger2.write('\nParameters: %d K' %(netParams(model)))
        logger2.write('\n%s\t\t%s\t\t%s\t\t%s\t\t%s'
                     % ('Epoch', 'TP', 'FP', 'TN', 'FN'))
    logger1.flush()
    logger2.flush()
    return logger1, logger2


def netParams(model):
    total_parameters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_parameters += p
    return total_parameters
