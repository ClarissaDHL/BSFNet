import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import cv2
import pickle

from PIL import Image
import matplotlib.pyplot as plt

from codes import cfg

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

palette = [0, 0, 0,255,0,0]

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_pascal_labels():
    return np.asarray([[0,0,0],[0,0,255]])

def decode_segmap(temp, plot=False):

    label_colours = get_pascal_labels()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 2):
        r[temp == l] = label_colours[l, 0]
        g[temp == l] = label_colours[l, 1]
        b[temp == l] = label_colours[l, 2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    #rgb = np.resize(rgb,(321,321,3))
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

#### source dataset is only avaiable by sending an email request to author #####
#### upon request is shown in http://ghsi.github.io/proj/RSS2016.html ####
#### more details are presented in http://ghsi.github.io/assets/pdfs/alcantarilla16rss.pdf ###
FP_MODIFIER=100

class CDD_Dataset(Dataset):

    def __init__(self,data_path, data_txt_path, crop_size=(256,256),
                 img1_mean=(128,128,128), img2_mean=(128,128,128), transform=True, transform_med = None):

        self.data_path = data_path
        self.data_txt_path = data_txt_path
        self.crop_h, self.crop_w = crop_size
        self.transform = transform
        self.transform_med = transform_med
        self.img1_mean = img1_mean
        self.img2_mean = img2_mean

        self.img_ids = [i_id.strip() for i_id in open(self.data_txt_path)]

        self.files = []

        for name in self.img_ids:
            img1_file = os.path.join(self.data_path, name.split()[0])
            img2_file = os.path.join(self.data_path, name.split()[1])
            label_file = os.path.join(self.data_path, name.split()[2])
            img_name = name.strip().split()[0].strip().split('.')[0].split('/')[-1]

            self.files.append({
                "img1": img1_file,
                "img2": img2_file,
                "label": label_file,
                "name": img_name
            })
        print('length of train set:',len(self.files))

    def data_transform(self, img1,img2,lbl):
        img1 = img1[:, :, ::-1]  # RGB -> BGR
        img1 = img1.astype(np.float64)
        img1 -= self.img1_mean
        img1 = img1.transpose(2, 0, 1)
        img1 = torch.from_numpy(img1).float()
        img2 = img2[:, :, ::-1]  # RGB -> BGR
        img2 = img2.astype(np.float64)
        img2 -= self.img2_mean
        img2 = img2.transpose(2, 0, 1)
        img2 = torch.from_numpy(img2).float()
        lbl = torch.from_numpy(lbl).long()
        return img1,img2,lbl

    def __getitem__(self, index):
        datafiles = self.files[index]

        ####### load images #############
        img1 = Image.open(datafiles['img1'])
        img2 = Image.open(datafiles['img2'])
        height,width,_ = np.array(img1,dtype= np.uint8).shape
        name = datafiles['name']
        if self.transform_med != None:
           img1 = self.transform_med(img1)
           img2 = self.transform_med(img2)
        img1 = np.array(img1,dtype= np.uint8)
        img2 = np.array(img2,dtype= np.uint8)

        ####### load labels ############
        label = Image.open(datafiles['label'])
        if self.transform_med != None:
            label = self.transform_med(label)
        label = np.array(label,dtype=np.int32)

        if self.transform:
            img1,img2,label = self.data_transform(img1,img2,label)

        return img1, img2, label, name, int(height), int(width)


    def __len__(self):
        return len(self.files)

class DataInform():
    def __init__(self, data_path='', classes=2, trainset_file='',
                 inform_data_file='', normVal=1.10):
        """
        Args:
           data_path: directory where the dataset is kept
           classes: number of classes in the dataset
           inform_data_file: location where cached file has to be stored
           normVal: normalization value, as defined in ERFNet paper
        """
        self.data_path = data_path
        self.classes = classes
        self.classweights = np.ones(self.classes, dtype=np.float32)
        self.weights = np.ones(self.classes, dtype=np.float32)
        self.normVal = normVal
        self.img1_mean = np.zeros(3, dtype=np.float32)
        self.img2_mean = np.zeros(3, dtype=np.float32)
        self.img1_std = np.zeros(3, dtype=np.float32)
        self.img2_std = np.zeros(3, dtype=np.float32)
        self.train_set_file = trainset_file
        self.inform_data_file = inform_data_file

    def compute_class_weights(self, histogram):
        """to compute the class weights
        Args:
            histogram: distribution of class samples
        """
        normHist = histogram / np.sum(histogram)
        for i in range(self.classes):
            self.classweights[i] = 1 / (np.log(self.normVal + normHist[i]))

    def readTrainSet(self, filename, train_flag=True):
        """to read the whole train set of current dataset.
         Args:
         fileName: train set file that stores the image locations
         trainStg: if processing training or validation data

         return: 0 if successful
         """
        global_hist = np.zeros(self.classes, dtype=np.float32)
        no_files = 0
        min_val_al = 0
        max_val_al = 0
        with open(os.path.join(self.data_path, filename), 'r') as textFile:
            for line in textFile:
                # we expect the text file to contain the data in following format
                # <RGB Image> <Label Image>
                line_arr = line.split()
                # img_file = ((self.data_path).strip() + '/' + line_arr[0].strip()).strip() #linux
                # label_file = ((self.data_path).strip() + '/' + line_arr[1].strip()).strip()
                img1_file = ((self.data_path).strip() + '/' + line_arr[0].strip()).strip()
                img2_file = ((self.data_path).strip() + '/' + line_arr[1].strip()).strip()
                label_file = ((self.data_path).strip() + '/' + line_arr[2].strip()).strip()

                n_pix = 0
                true_pix = 0

                label_img = cv2.imread(label_file, 0)
                unique_values = np.unique(label_img)
                max_val = max(unique_values)
                min_val = min(unique_values)

                max_val_al = max(max_val_al, max_val)
                min_val_al = min(min_val_al, min_val)

                label = np.array(label_img, dtype=np.int32)
                label_norm = label
                label_norm[label > 0] = 1
                label_shape = label.shape
                n_pix += np.prod(label_shape)
                true_pix += label_norm.sum()
                self.weights = [FP_MODIFIER * 2 * true_pix / n_pix, 2 * (n_pix - true_pix) / n_pix]

                if train_flag == True:
                    hist = np.histogram(label_img, self.classes, [0, self.classes - 1])
                    global_hist += hist[0]

                    rgb_img1 = cv2.imread(img1_file)
                    self.img1_mean[0] += np.mean(rgb_img1[:, :, 0])
                    self.img1_mean[1] += np.mean(rgb_img1[:, :, 1])
                    self.img1_mean[2] += np.mean(rgb_img1[:, :, 2])

                    self.img1_std[0] += np.std(rgb_img1[:, :, 0])
                    self.img1_std[1] += np.std(rgb_img1[:, :, 1])
                    self.img1_std[2] += np.std(rgb_img1[:, :, 2])

                    rgb_img2 = cv2.imread(img2_file)
                    self.img2_mean[0] += np.mean(rgb_img2[:, :, 0])
                    self.img2_mean[1] += np.mean(rgb_img2[:, :, 1])
                    self.img2_mean[2] += np.mean(rgb_img2[:, :, 2])

                    self.img2_std[0] += np.std(rgb_img2[:, :, 0])
                    self.img2_std[1] += np.std(rgb_img2[:, :, 1])
                    self.img2_std[2] += np.std(rgb_img2[:, :, 2])
                else:
                    print('we can only get statistic information of train set')
                if max_val > (self.classes - 1) or min_val < 0:
                    print('Labels can take value between 0 and the number of classes')
                    print('Some problems with labels. Please check label_set:', unique_values)
                    print('Label Img ID:' + label_file)
                no_files += 1

        self.img1_mean /= no_files
        self.img1_std /= no_files
        self.img2_mean /= no_files
        self.img2_std /= no_files

        self.compute_class_weights(global_hist)
        return 0

    def CollectDataAndSave(self):
        """ To collect statistical information of train set and then save it.
        The file train.txt should be inside the data directory.
        """
        print('Processing training data')
        return_val = self.readTrainSet(filename=self.train_set_file)

        print('Saving data')
        if return_val == 0:
            data_dict = dict()
            data_dict['img1_mean'] = self.img1_mean
            data_dict['img1_std'] = self.img1_std
            data_dict['img2_mean'] = self.img2_mean
            data_dict['img2_std'] = self.img2_std
            data_dict['classweights'] = self.classweights
            data_dict['weights'] = self.weights
            pickle.dump(data_dict, open(self.inform_data_file, 'wb'))
            return data_dict
        return None

def main():
    data_root = cfg.DATA_PATH
    train_list = cfg.TRAIN_TXT_PATH
    val_list = cfg.VAL_TXT_PATH
    trainval_list = cfg.TRAINVAL_TXT_PATH
    inform_data_file = os.path.join('./inform', 'inform1.pkl')

    train_data = CDD_Dataset(data_path=data_root, data_txt_path=train_list)
    dataCollect = DataInform(data_root, 2, trainval_list, inform_data_file)
    datas = dataCollect.CollectDataAndSave()

    print('Dataset Statistics')
    print('data[classweight]:', datas['classweights'])
    print('data[weight]:', datas['weights'])
    print('img1_mean and img1_std:', datas['img1_mean'], datas['img1_std'])
    print('img2_mean and img2_std:', datas['img2_mean'], datas['img2_std'])

