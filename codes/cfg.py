import os

BASE_PATH = '/home/dhl/PycharmProjects/BSFNet_upload'

DATA_PATH = '/home/dhl/PycharmProjects/CDD'
TRAINVAL_TXT_PATH = os.path.join(DATA_PATH, 'trainval.txt')
TRAINTEST_TXT_PATH = os.path.join(DATA_PATH, 'traintest.txt')
TRAIN_TXT_PATH = os.path.join(DATA_PATH, 'train.txt')
VAL_TXT_PATH = os.path.join(DATA_PATH, 'val.txt')
TEST_TXT_PATH = os.path.join(DATA_PATH, 'test.txt')
INFORM_PATH = os.path.join(BASE_PATH, 'data', 'inform')

################Siamese Neural Network parameters#############
SAVE_PATH = os.path.join(BASE_PATH, 'results')


DECAY = 5e-5
b1 = 0.50 #adam beta1:0.5 or 0.9
b2 = 0.999
MAX_ITER = 40000
BATCH_SIZE = 4

INPUT_SIZE = (256, 256)
num_workers = 4

