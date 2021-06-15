import json
import os
import copy

with open('../infos/directory.json') as fp: all_data_dir = json.load(fp)

OBJ_PATH_train_s = all_data_dir + 'Object_Detections_hico_orig/train/'
OBJ_PATH_test_s = all_data_dir + 'Object_Detections_hico/test/'


for i in range(len(os.listdir(OBJ_PATH_train_s))):


    try:
        with open(OBJ_PATH_train_s + os.listdir(OBJ_PATH_train_s)[i]) as fp:
            origin_annotation = json.load(fp)

    except Exception as ex:
        import pdb; pdb.set_trace()

    origin_h = origin_annotation['H']
    origin_w = origin_annotation['W']
    origin_detections = origin_annotation['detections']
    origin_filename = origin_annotation['image_name']


    for j in range(len(origin_detections)):

        q2_tmp = copy.deepcopy(origin_detections[j])
        q3_tmp = copy.deepcopy(origin_detections[j])
        q4_tmp = copy.deepcopy(origin_detections[j])

        q2_tmp['box_coords'][0] = q2_tmp['box_coords'][0]
        q2_tmp['box_coords'][1] = q2_tmp['box_coords'][1] + origin_w
        q2_tmp['box_coords'][2] = q2_tmp['box_coords'][2]
        q2_tmp['box_coords'][3] = q2_tmp['box_coords'][3] + origin_w

        q3_tmp['box_coords'][0] = q3_tmp['box_coords'][0] + origin_h
        q3_tmp['box_coords'][1] = q3_tmp['box_coords'][1]
        q3_tmp['box_coords'][2] = q3_tmp['box_coords'][2] + origin_h
        q3_tmp['box_coords'][3] = q3_tmp['box_coords'][3]

        q4_tmp['box_coords'][0] = q4_tmp['box_coords'][0] + origin_h
        q4_tmp['box_coords'][1] = q4_tmp['box_coords'][1] + origin_w
        q4_tmp['box_coords'][2] = q4_tmp['box_coords'][2] + origin_h
        q4_tmp['box_coords'][3] = q4_tmp['box_coords'][3] + origin_w


        origin_annotation['detections'].append(q2_tmp)
        origin_annotation['detections'].append(q3_tmp)
        origin_annotation['detections'].append(q4_tmp)


    with open(all_data_dir + 'Object_Detections_hico/train/' + origin_filename.split('.')[0] + '.json', 'w') as write_json:
        json.dump(origin_detections, write_json)









