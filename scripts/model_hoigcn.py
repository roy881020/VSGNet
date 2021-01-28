from __future__ import print_function, division
import torch
import torch.nn as nn

import os
import numpy as np
import pool_pairing  as ROI

import torchvision.models as models

lin_size = 1024
ids = 80
context_size = 1024
sp_size = 1024
mul = 3
deep = 512
pool_size = (10, 10)
pool_size_pose = (18, 5, 5)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)


class VSGNet(nn.Module):
    def __init__(self):
        super(VSGNet, self).__init__()

        model = models.resnet152(pretrained=True)
        self.flat = Flatten()

        self.Conv_pretrain = nn.Sequential(*list(model.children())[0:7])  ## Resnets,resnext


        self.sigmoid = nn.Sigmoid()


        self.learnable_conv = nn.Sequential(
            nn.Conv1d(3072, 1024, kernel_size=1, stride=1),
            nn.BatchNorm1d(1024),
            nn.Conv1d(1024, 512, kernel_size=1, stride=1),
            nn.BatchNorm1d(512),
            nn.Conv1d(512,1024,kernel_size=1, stride=1),
            nn.BatchNorm1d(1024),
            nn.Conv1d(1024,3072,kernel_size=1, stride=1),
            nn.BatchNorm1d(3072),
            nn.ReLU(inplace=False),
        )
        self.learnable_single = nn.Sequential(
            nn.Linear(lin_size * 3, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 128),
            nn.Linear(128, 1),
            nn.ReLU(),
        )
        self.learnable_matrix = nn.Sequential(
            nn.Linear(lin_size * 3, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 128),
            nn.Linear(128, 29),
            nn.ReLU(),
        )
        self.interaction_prob_matrix = nn.Sequential(
            nn.Linear(lin_size * 3, 1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.interaction_prob_value = nn.Sequential(
            nn.Linear(512, 128),
            nn.Linear(128,29),
        )

        self.test_spatial_to_node_feature = nn.Sequential(
            nn.Linear(3076,3072),
            nn.ReLU()
        )

    def forward(self, x, pairs_info, pairs_info_augmented, image_id, flag_, phase):
        out1 = self.Conv_pretrain(x)  ### out1_shape(batch, 1024, 25, 25) -> (batch, channel, width, height) from (batch, 3, 400, 400)

        import pdb; pdb.set_trace()

        rois_people, rois_objects, spatial_locs, union_box = ROI.get_pool_loc(out1, image_id, flag_, size=pool_size,
                                                                              spatial_scale=25,
                                                                              batch_size=len(pairs_info))
        #rois_people -> total batch, person area, calculate each size after then, adaptive pool to make same size (10, 10)
        #rois_objects -> same with rois_people in place of people
        #spatial_locs -> objects(persons~, objects~) [x1, y1, x2, y2, x2-x1, y2-y1] , scale 0~25 from spatial_scale option
        #union_box -> in each batch, every person and objects combination spatial map that value is 100, ohter area is 0


        ### Defining The Pooling Operations #######
        x, y = out1.size()[2], out1.size()[3]
        hum_pool = nn.AvgPool2d(pool_size, padding=0, stride=(1, 1))
        obj_pool = nn.AvgPool2d(pool_size, padding=0, stride=(1, 1))
        context_pool = nn.AvgPool2d((x, y), padding=0, stride=(1, 1))
        #################################################
        ### Human###
        residual_people = rois_people
        res_people = self.Conv_people(rois_people) + residual_people
        res_av_people = hum_pool(res_people)
        out2_people = self.flat(res_av_people)
        ###########

        ##Objects##
        residual_objects = rois_objects
        res_objects = self.Conv_objects(rois_objects) + residual_objects
        res_av_objects = obj_pool(res_objects)
        out2_objects = self.flat(res_av_objects)
        #############

        #### Context ######
        residual_context = out1
        res_context = self.Conv_context(out1) + residual_context
        res_av_context = context_pool(res_context)
        out2_context = self.flat(res_av_context)
        #################

        ##Attention Features##
        out2_union = self.spmap_up(self.flat(self.conv_sp_map(union_box)))
        ############################



        pairs, people, objects_only = ROI.pairing(out2_people, out2_objects, out2_context, spatial_locs, pairs_info)

        pairs = self.test_spatial_to_node_feature(pairs)

        node_feature_cat = ROI.get_node_feature(out2_people, out2_objects, out2_context, pairs_info)
        node_feature = node_feature_cat.reshape(node_feature_cat.size()[0], node_feature_cat.size()[1], 1)
        #node_feature = self.learnable_conv(node_feature)
        node_feature = node_feature.reshape(node_feature.size()[0], node_feature.size()[1])

        ###210121 test spatial_locs feature into node_feature by ADD
        ##after this test, have to test by MUL
        #import pdb; pdb.set_trace()
        node_feature = node_feature * pairs

        test = self.learnable_matrix(node_feature)
        # test2 = self.learnable_conv(node_feature.unsqueeze(2).unsqueeze(2))
        # test2 = self.learnable_conv_matrix(test2)
        test2 = self.learnable_single(node_feature)

        ###Interaction Prob
        interaction_feature = self.interaction_prob_matrix(node_feature_cat)
        interaction_prob = interaction_feature * out2_union
        interaction_score = self.interaction_prob_value(interaction_prob)
        interaction_score = self.sigmoid(interaction_score)



        return [test, test2, interaction_score] # ,lin_obj_ids]
