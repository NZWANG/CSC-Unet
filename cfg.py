# -*- encoding: utf-8 -*-
# Author  : Haitong

EPOCH_NUMBER = 200
BATCH_SIZE = 4
DATASET = ['CamVid', 12]
crop_size = (352, 480)
unfolding = 2
lr = 1e-4

class_dict_path = './Datasets/' + DATASET[0] + '/class_dict.csv'
TRAIN_ROOT = './Datasets/' + DATASET[0] + '/train'
TRAIN_LABEL = './Datasets/' + DATASET[0] + '/train_labels'
VAL_ROOT = './Datasets/' + DATASET[0] + '/valid'
VAL_LABEL = './Datasets/' + DATASET[0] + '/valid_labels'
TEST_ROOT = './Datasets/' + DATASET[0] + '/test'
TEST_LABEL = './Datasets/' + DATASET[0] + '/test_labels'
