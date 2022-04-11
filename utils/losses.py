import torch
import numpy as np


def distance_loss(output,target,masks):
    dist_loss = torch.zeros(1,dtype=torch.float,requires_grad=True).to(target[0].device)
    for sent_embs, gold_matrix, mask_ixs in zip(output,target,masks):
        #assert sent_embs.shape[0] == gold_matrix.shape[0] == mask.shape[0], "sentence hidden state shape | gold distance matrix shape | punctuation mask shape NOT MATCHED."
        tmp_length = len(gold_matrix)
        pred_matrix = ((sent_embs.unsqueeze(1)-sent_embs.unsqueeze(0))**2).sum(-1)
        # keep only elements at positions except mask_ixs;
        # use only upper triangle part of the matrix
        mask = torch.triu(torch.ones(tmp_length,tmp_length),1).to(output.get_device())
        mask[mask_ixs,:] = 0
        mask[:,mask_ixs] = 0
        real_length = mask.sum().item()
        assert pred_matrix.shape[0] == pred_matrix.shape[1] == tmp_length, "predicted distance matrix not in good shape."
        real_length = mask.eq(0).sum().item() # length of token sequences with punctuations removed
        #calculate the loss after removing punctuations + [CLS] & [SEP]
        dist_loss += torch.abs(pred_matrix - gold_matrix**2).masked_fill_(mask.eq(0),0).sum() / real_length
    return dist_loss / len(target)

def depth_loss(output,target,masks):
    depth_loss = torch.zeros(1,dtype=torch.float,requires_grad=True).to(target[0].device)
    for sent_embs, gold_depths, mask_ixs in zip(output,target,masks):
        #assert sent_embs.shape[0] == gold_depths.shape[0] == mask.shape[0], "sentence hidden state shape | gold depth shape | punctuation mask shape NOT MATCHED."
        tmp_length = len(gold_depths)
        pred_depths = (sent_embs**2).sum(-1)
        mask = torch.ones(tmp_length).to(output.get_device())
        assert pred_depths.shape[0] == tmp_length, "predicted depths not in good shape."
        mask[mask_ixs] = 0
        real_length = mask.sum().item()
        depth_loss += torch.abs(pred_depths - gold_depths**2).masked_fill_(mask.eq(0),0).sum() / real_length
    return depth_loss / len(target)

