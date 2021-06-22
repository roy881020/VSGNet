import json
import os
import copy

with open('../All_data/Annotations_hico/train_annotations_quattro.json') as fp:
    hoi_data = json.load(fp)

with open('../All_data/Object_Detections_hico/train_quattro') as fp:
    hoi_data = json.load(fp)

import pdb; pdb.set_trace()
