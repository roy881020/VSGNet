import torch
import torch.nn as nn
import time
import errno
import os
import gc
import pickle
import shutil
import json
import os
import pandas as pd
from skimage import io, transform
import numpy as np
import calculate_ap_classwise as ap
import matplotlib.pyplot as plt
import random

import helpers_preprocess as helpers_pre
import pred_vis as viss
import prior_vcoco as prior
import proper_inferance_file as proper
from tqdm import tqdm
import pdb

sigmoid = nn.Sigmoid()

### Fixing Seeds#######
device = torch.device("cuda")
seed = 10
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
softmax = nn.Softmax()
##########################################


########## Paramets for person to object class mapping#####
SCORE_TH = 0.6
SCORE_OBJ = 0.3
epoch_to_change = 400
thresh_hold = -1
##############################################################

#### Loss functions Definition########
loss_com = nn.BCEWithLogitsLoss(reduction='sum')
loss_com_class = nn.BCEWithLogitsLoss(reduction='none')
loss_com_combine = nn.BCELoss(reduction='none')
loss_com_single = nn.BCEWithLogitsLoss(reduction='sum')
loss_multilabel = nn.BCELoss(reduction='sum')
loss_Regression = nn.MSELoss()
##################################
# import pdb;pdb.set_trace()
no_of_classes = 29



#### Saving CheckPoint##########
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def train_test(model, optimizer, scheduler, dataloader, number_of_epochs, break_point, saving_epoch, folder_name,
               batch_size, infr, start_epoch, mean_best, visualize):

    torch.cuda.empty_cache()
    phases =['train','val', 'test']
    end_epoch = start_epoch + number_of_epochs
    test_set = np.zeros((1,80))
    test_set[0][0] = 1
    test_set[0][37] = 1
    test_set = torch.tensor(test_set).float()

    for epoch in range(start_epoch, end_epoch):
        scheduler.step()
        print('Epoch {}/{}'.format(epoch + 1, end_epoch))
        print('-' * 10)
        initial_time_epoch = time.time()

        for phase in phases:
            if phase == 'train':
                model.train()
            elif phase == 'val':
                model.train()
            else:
                model.eval()

            print('In {}'.format(phase))

            for iterr, data in enumerate(tqdm(dataloader[phase])):
                if iterr % 20 == 0:
                    torch.cuda.empty_cache()

                image_id = data[0].to(device) #batch_size of image_id, consist of tensor
                action_labels = data[1].to(device) #batch_size of action labels, consist of tensor
                inputs_object_list = helpers_pre.get_object_list(image_id, phase).to(device) #batch size of input object list (batch, 80)
                inputs_object_list = inputs_object_list.float()

                true = action_labels.data.cpu().numpy()

                ######test
                softmax = nn.Softmax()
                action_labels = softmax(action_labels.float())
                ####

                with torch.set_grad_enabled(phase == 'train' or phase == 'val'):

                    out = model(inputs_object_list, phase)

                    #import pdb; pdb.set_trace()

                    loss_result = loss_multilabel(out, action_labels)

                    # if loss_result.item() <= 0:
                    #     import pdb; pdb.set_trace()

                    loss_this_epoch = torch.sum(loss_result).item()

                    if phase =='train' or phase =='val':
                        loss_result.backward()
                        optimizer.step()

                    del loss_result
                    del out
                    del inputs_object_list
                    del action_labels

            if phase == 'train':
                print('train loss :{}'.format(loss_this_epoch))
            elif phase == 'val':
                print('val loss :{}'.format(loss_this_epoch))
            else:
                print('test loss :{}'.format(loss_this_epoch))
            if epoch == 900:
                result = model(test_set, 'test')
                print(result)
                import pdb;pdb.set_trace()
            if (epoch + 1) % saving_epoch == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, filename=folder_name + '/' +str(epoch+1) + 'checkpoint.pth.tar')

    return
