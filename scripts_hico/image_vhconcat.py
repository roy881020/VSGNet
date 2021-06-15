import json
import os
import copy
import cv2

with open('../infos/directory.json') as fp: all_data_dir = json.load(fp)

OBJ_PATH_train_s = all_data_dir + 'Object_Detections_hico_orig/train/'
OBJ_PATH_test_s = all_data_dir + 'Object_Detections_hico/test/'
IMG_PATH_train = all_data_dir + 'Data_hico/train2015_origin/'
TARGET_PATH_train = all_data_dir + 'Data_hico/train2015/'

for i in range(len(os.listdir(IMG_PATH_train))):
    with open(IMG_PATH_train + os.listdir(IMG_PATH_train)[i]) as fp:
        img = cv2.imread(fp)

    quattro_img = cv2.vconcat([img, img])
    quattro_img = cv2.hconcat([quattro_img, quattro_img])

    cv2.imwrite(TARGET_PATH_train + os.listdir(IMG_PATH_train)[i], quattro_img)

    print('%d img saved'.format(os.listdir(IMG_PATH_train)[i]))






