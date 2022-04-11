import os
import re
import numpy as np
import pandas as pd
import logging
import torch
import pickle
from sklearn.preprocessing import MultiLabelBinarizer

#lb = MultiLabelBinarizer(classes=[0,3,4,5,6,9])

#m = torch.nn.Softmax(dim=1)
mlb = MultiLabelBinarizer(classes=list(range(14)))
logger = logging.getLogger(__name__)

class DataLoader(object):
    def __init__(self,path,name,mode,seed,batch_size,device,debug):
        np.random.seed(seed)
        self.device = device
        self.mode = mode
        self.name = name
        data = pickle.load(open(os.path.join(path,f"{name}.pkl"),"rb"))
        data = [list(d.values()) for d in data]

        num_examples = len(data)

        #print(debug)
        if name == "train":
            # shuffle on train
            np.random.shuffle(data)
        
        if debug:
            data = data[:100]
            num_examples = 100
        self.data = data
        
        logger.info(len(self.data))
        self.data = [self.data[i:i+batch_size] for i in range(0,num_examples,batch_size)] 
        logger.info(f"{name}: {len(self.data)} batches generated.")
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self,key):
        if not isinstance(key,int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        unzip_batch = map(list,zip(*batch))
        # 'wp_ids', 'map', 'distance_matrix', 'depth_list', 'head_indexes', 'mask_indexes', 'subj_range', 'obj_range', 'label'
        
        wp_ids = unzip_batch[0]
        max_length = max(map(len,wp_ids))
        wp_ids = torch.Tensor([wp + [0] * (max_length-len(wp)) for wp in wp_ids]).long().to(self.device)
        att_masks = torch.Tensor([len(line) * [1] + (max_length-len(line)) * [0] for line in wp_ids]).int().to(self.device)
       
        assert len(unzip_batch) == 9, "mode WITH_SYNTAX and input data do not match."
        if self.mode == "no_syntax":
            encoding =  {"wp_ids":wp_ids,
                         "attention_mask":att_masks,
                         "subj_range":unzip_batch[-3],
                         "obj_range":unzip_batch[-2]}
        elif self.mode == "with_syntax":
            encoding = {"wp_ids":wp_ids,
                        "attention_mask":att_masks,
                        "maps":[list(d.values()) for d in unzip_batch[1]],
                        "dist_matrixs":[torch.Tensor(lst).int().to(self.device) for lst in unzip_batch[2]],
                        "depths":[torch.Tensor(lst).int().to(self.device) for lst in unzip_batch[3]],
                        "heads":[torch.Tensor(h).int().to(self.device) for h in unzip_batch[4]],
                        "masks":unzip_batch[5],
                        "subj_range":unzip_batch[-3],
                        "obj_range":unzip_batch[-2]}

        if self.name == "test":
            return encoding
        else:
            encoding.update({"labels":torch.Tensor(unzip_batch[-1]).long().to(self.device)})
            return encoding

    def _rep_deps(self,dep_locs):
        replicated_dep_locs = []
        for dl in dep_locs:
            tmp_locs = []
            for i1, i2 in dl:
                tmp_locs.append((i1,i2))
                tmp_locs.append((i2,i1))
            replicated_dep_locs.append(tmp_locs)
        return replicated_dep_locs

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
