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
    def __init__(self,path,name,seed,batch_size,device,debug):
        np.random.seed(seed)
        self.device = device
        self.name = name
        data = pickle.load(open(os.path.join(path,f"{name}.pkl"),"rb"))

        wp_ids = data["wp_ids"]
        maps = data["maps"]
        dist_matrices = data["distance_matrices"]
        depths = data["depths"]
        masks = data["masks"]
        keys = data["keys"]
        labels = data["labels"]
        dep_locs = data["dep_locs"]
        num_examples = len(labels)

        #print(debug)
        if name == "train":
            # shuffle on train
            indexes = list(range(num_examples))
            np.random.shuffle(indexes)
            wp_ids = [wp_ids[i] for i in indexes]
            labels = [labels[i] for i in indexes]
            maps = [maps[i] for i in indexes]
            dist_matrices = [dist_matrices[i] for i in indexes]
            depths = [depths[i] for i in indexes]
            masks = [masks[i] for i in indexes]
            keys = [keys[i] for i in indexes]
            dep_locs = [dep_locs[i] for i in indexes]

        # order: wordpiece_ids, maps, keys, masks, dist_matrix, depth_list,relation_labels, 
        if debug:
            self.data = list(zip(wp_ids,maps,keys,masks,dep_locs,dist_matrices,depths,labels))[:100]
            num_examples = 100
        else:
            self.data = list(zip(wp_ids,maps,keys,masks,dep_locs,dist_matrices,depths,labels))
        
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
        unzip_batch = list(zip(*batch))
        
        wp_ids = unzip_batch[0]
        max_length = max(map(len,wp_ids))
        wp_ids = torch.Tensor([wp + [0] * (max_length-len(wp)) for wp in wp_ids]).long().to(self.device)
        att_masks = torch.Tensor([len(line) * [1] + (max_length-len(line)) * [0] for line in wp_ids]).int().to(self.device)
       
        #print(unzip_batch[4])
        #print(mlb.fit_transform(unzip_batch[4])) 
        assert len(unzip_batch) == 8, "mode WITH_SYNTAX and input data do not match."
        encoding = {"wps":wp_ids,
                    "attention_mask":att_masks,
                    "maps":unzip_batch[1],
                    "keys":[torch.Tensor(lst).int().to(self.device) for lst in unzip_batch[2]],
                    "masks":unzip_batch[3],
                    "dep_locs":self._rep_deps(unzip_batch[4]),
                    "dist_matrixs":[torch.Tensor(lst).int().to(self.device) for lst in unzip_batch[5]],
                    "depths":[torch.Tensor(lst).int().to(self.device) for lst in unzip_batch[6]],
                    }

        if self.name == "test":
            return encoding
        else:
            encoding.update({"labels":torch.Tensor(mlb.fit_transform(unzip_batch[7])).float().to(self.device)})
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
