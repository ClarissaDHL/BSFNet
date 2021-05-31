import os
import pickle
from torch.utils import data
from data.CDD import *
from data import LEVIR_slice as slc

# from codes import LEVIR_cfg as cfg
from codes import cfg

def train_dataset(dataset, crop_size, batch_size, num_workers):
    data_path = cfg.DATA_PATH
    trainval_list = cfg.TRAINVAL_TXT_PATH
    traintest_list = cfg.TRAINTEST_TXT_PATH
    train_data_list = cfg.TRAIN_TXT_PATH
    val_data_list = cfg.VAL_TXT_PATH
    test_data_list = cfg.TEST_TXT_PATH
    # inform_data_file = os.path.join(cfg.INFORM_PATH, dataset + '_inform_slice.pkl')
    inform_data_file = os.path.join(cfg.INFORM_PATH, dataset + '_inform_trainval.pkl')

    flag = os.path.isfile(inform_data_file)
    if not flag:
        print('%s is not found' %(inform_data_file))
        dataCollect = DataInform(data_path, 2, trainval_list,inform_data_file)
        datas = dataCollect.CollectDataAndSave()
        if datas is None:
            print('Error when pickling data, please check')
            exit(-1)
    else:
        print('find file:', str(inform_data_file))
        datas = pickle.load(open(inform_data_file, 'rb'))

    if dataset == 'CDD':
        trainloader = data.DataLoader(
            CDD_Dataset(data_path, train_data_list, crop_size=crop_size,
                        img1_mean=datas['img1_mean'], img2_mean=datas['img2_mean']),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)
        valloader = data.DataLoader(
            CDD_Dataset(data_path, test_data_list, crop_size=crop_size,
                        img1_mean=datas['img1_mean'], img2_mean=datas['img2_mean']),
            batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    elif dataset == 'LEVIR':
        trainloader = data.DataLoader(
            slc.LEVIR_Dataset(data_path, train_data_list, crop_size=crop_size,
                            img1_mean=datas['img1_mean'], img2_mean=datas['img2_mean']),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)
        valloader = data.DataLoader(
            slc.LEVIR_Dataset(data_path, val_data_list, crop_size=crop_size,
                            img1_mean=datas['img1_mean'], img2_mean=datas['img2_mean']),
            batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)


    return datas, trainloader, valloader

if __name__ == '__main__':
    dataset = 'CDD'
    crop_size = (256,256)
    classes = 2
    batch_size = 4
    num_workers = 4

    datas, trainloader, valloader = train_dataset(dataset, crop_size, batch_size, num_workers)
    print('dataset: %s' %(dataset))
    print('Dataset Statistics')
    print('data[classweight]:', datas['classweights'])
    print('data[weight]:', datas['weights'])
    print('img1_mean and img1_std:', datas['img1_mean'], datas['img1_std'])
    print('img2_mean and img2_std:', datas['img2_mean'], datas['img2_std'])


