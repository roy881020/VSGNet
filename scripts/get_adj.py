import numpy as np
import pickle

with open('../infos/prior.pickle', 'rb') as fp: priors = pickle.load(fp, encoding='bytes')


for i in range(29):
    count = 0
    for j in range(80):
        if(priors[j+1][0][i]):
            count += 1
    print("{}'s action has {} count".format(i + 1, count))
    count = 0

import pdb; pdb.set_trace()