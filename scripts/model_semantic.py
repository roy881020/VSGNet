import torch
import torch.nn as nn
import numpy as np
import torchvision
import helpers_preprocess as pre

NUM_ACTION = 29
NUM_OBJECTS = 80

class Semantic(nn.Module):
    def __init__(self):
        super(Semantic, self).__init__()

        self.sigmoid = nn.Sigmoid()

        self.hidden_unit = nn.Sequential(
            nn.Linear(NUM_OBJECTS, 40, bias=True),
            nn.Sigmoid(),
            nn.Linear(40, 40, bias=True),
            nn.Sigmoid(),
            nn.Linear(40, 40, bias=True),
            nn.Sigmoid(),
            nn.Linear(40, 40, bias=True),
            nn.Sigmoid(),
            nn.Linear(40, 29, bias=True),
            nn.Sigmoid(),
        )

        self.hidden_unit_relu = nn.Sequential(
            nn.Linear(NUM_OBJECTS, 40, bias=True),
            nn.ReLU(),
            nn.Linear(40, 40, bias=True),
            nn.ReLU(),
            nn.Linear(40, 40, bias=True),
            nn.ReLU(),
            nn.Linear(40, 40, bias=True),
            nn.ReLU(),
            nn.Linear(40, 29, bias=True),
            nn.ReLU(),
        )

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, input_object, flag):

        #predicted_label = self.hidden_unit(input_object)
        predicted_label = self.hidden_unit_relu(input_object)
        #import pdb; pdb.set_trace()

        predicted_label = self.softmax(predicted_label)

        return predicted_label