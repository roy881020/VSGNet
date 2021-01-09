import torch
import torch.nn as nn
import numpy as np
import pdb
import model_hoigcn as mh

device = torch.device("cuda")

############fake dataset##############
fake_inputs = np.ones((2,3,400,400))
fake_inputs = torch.from_numpy(fake_inputs)
fake_inputs = fake_inputs.float()

fake_pairs_info = [[2, 10, 29], [4, 20, 29]]
fake_pairs_info = torch.LongTensor(fake_pairs_info)

fake_image_id = [41135]
fake_image_id = torch.tensor(fake_image_id)

fake_flag = [[0,0]]
fake_flag = torch.tensor(fake_flag)

fake_phase = 'train'
##################################3

test_model = mh.VSGNet()
test_model = nn.DataParallel(test_model)
test_model.to(device)


out = test_model(fake_inputs, fake_pairs_info, fake_pairs_info, fake_image_id, fake_flag, fake_phase)




pdb.set_trace()


