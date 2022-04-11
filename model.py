from math import dist
import numpy as np
import random
import logging
import torch
from torch import nn
from torch.nn import (BCEWithLogitsLoss,CrossEntropyLoss)
from transformers import (BertPreTrainedModel, BertModel)
from utils.losses import distance_loss, depth_loss
from utils.constant import class_weights, class_weights_dist, class_weights_depth, n_dist_cats, n_depth_cats

logger = logging.getLogger(__name__)

def set_seed(args,ensemble_id):
    seed = args.seed + ensemble_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(3*config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, subj_range, obj_range):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        sent_tensors = []
        for hidden_state, sr, obr in zip(hidden_states,subj_range,obj_range):
            first_token_tensor = hidden_state[0]
            subj_tensor = torch.mean(hidden_state[sr[0]:sr[1]],1)
            obj_tensor = torch.mean(hidden_state[obr[0]:obr[1]],1)
            sent_tensors.append(torch.cat([first_token_tensor,subj_tensor,obj_tensor]).unsqueeze(0))
        pooled_output = self.dense(sent_tensors)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class SyntaxBertModel(BertPreTrainedModel):
    def __init__(self,config,dataset_name,mode,num_labels=2,probe_rank=-1,target_layer_ix=12,mask_percentage=0.15,coef=1):
        super().__init__(config)
        self.num_labels = num_labels
        self.mode = mode
        self.coef = coef
        self.num_bert_layers = config.num_hidden_layers
        self.target_layer_ix = target_layer_ix
        self.mask_percentage = mask_percentage
        if probe_rank == -1:
            self.probe_rank = config.hidden_size
        else:
            self.probe_rank = probe_rank

        self.bert = BertModel(config)
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
       
        # fine-tuning weights
        if mode == "no_syntax":
            self.clf = nn.Linear(config.hidden_size,self.num_labels)
            self.loss_fct = CrossEntropyLoss(weight=torch.Tensor(np.log(class_weights[dataset_name]))) 
        elif mode == "with_syntax":
            self.trans_dist = nn.Linear(config.hidden_size,self.probe_rank)
            self.trans_depth = nn.Linear(config.hidden_size,self.probe_rank)
            #self.clf_dist = nn.Linear(self.probe_rank,n_dist_cats)
            #self.clf_depth = nn.Linear(self.probe_rank,n_depth_cats)
            #self.dist_loss_fct = CrossEntropyLoss(weight=torch.Tensor(np.log(class_weights_dist)),ignore_index=0)
            #self.depth_loss_fct = CrossEntropyLoss(weight=torch.Tensor(np.log(class_weights_depth)))         
            self.dist_loss_fct = distance_loss
            self.depth_loss_fct = depth_loss

        logger.info(f"SyntaxBERT loaded for dataset {dataset_name}: num of labels={num_labels}; probe dimension={self.probe_rank};"+\
                    f"mode={self.mode}; alpha={self.coef}")

        self.init_weights()

    def forward(self,
                wp_ids=None,
                maps=None,
                masks=None,
                heads=None,
                dist_matrixs=None,
                depths=None,
                subj_range=None,
                obj_range=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                labels=None,
                predict_only=False):
        outputs = self.bert(input_ids=wp_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            output_hidden_states=True)
        #sequence_output = self.dropout(outputs[2][self.target_layer_ix])
        sequence_output = outputs[0]
        pooled_output = self.pooler(sequence_output,subj_range,obj_range)
        logits = self.clf(self.dropout(pooled_output))

        if not predict_only:
            re_loss = self.loss_fct(logits.view(-1,self.num_labels),labels)
            if self.mode == "no_syntax":
                return (re_loss,logits)
            token_embs = self._get_token_embs(sequence_output,maps)
            token_logits_dist = self.trans_dist(token_embs)
            token_logits_depth = self.trans_depth(token_embs)
            # relation classification loss + distance probe loss + depth probe loss
            """logits_dist = []; logits_depth = []; labels_dist = []; labels_depth = []
            for embs_dist, embs_depth, dist_mat, depth_lst, index_dist, index_depth in zip(token_logits_dist,token_logits_depth,dist_matrixs,depths,
                                                                                            selection_indexes_dist,selection_indexes_depth):
                #tmp_len = dist_mat.shape[0]
                assert embs_dist.shape[0] == embs_depth.shape[0] == dist_mat.shape[0] == depth_lst.shape[0], "sequence length not matched for embeddings and targets."
                # pairwise difference of vi, vj
                preds_dist = self.clf_dist((embs_dist.unsqueeze(1) - embs_dist.unsqueeze(0)) ** 2)
                preds_depth = self.clf_depth(embs_depth ** 2)
                index_dist = list(zip(*index_dist))
                logits_dist.append(preds_dist[index_dist])
                logits_depth.append(preds_depth[index_depth])
                labels_dist.append(dist_mat[index_dist])
                labels_depth.append(depth_lst[index_depth])
            logits_dist = torch.cat(logits_dist,dim=0); logits_depth = torch.cat(logits_depth,dim=0)
            labels_dist = torch.cat(labels_dist,dim=0); labels_depth = torch.cat(labels_depth,dim=0)"""
            syntactic_loss = torch.sqrt(self.dist_loss_fct(token_logits_dist,dist_matrixs,masks) + self.depth_loss_fct(token_logits_depth,depths,masks))
            loss = (re_loss + self.coef * syntactic_loss) / (1 + self.coef)
            return (loss, logits, syntactic_loss.item()) 
        else:
            return (logits,) 

    """def _generate_syntactic_mask(self,masks,dep_locs):
        # [1,0,1] for mask means that the second token is a punctuation
        selection_indexes_dist = []
        selection_indexes_depth = []
        for msk, loc in zip(masks,dep_locs):
            tmp_len = len(msk)
            tmp_msk = torch.triu(torch.ones(tmp_len,tmp_len),diagonal=1)
            punc_loc = [i for i,v in enumerate(msk) if v == 0]
            tmp_msk[punc_loc,:] = 0
            tmp_msk[:,punc_loc] = 0
            tmp_msk[list(zip(*loc))] = 0
            tmp_msk = torch.Tensor(tmp_msk).to(next(self.parameters()).device)
            indexes = torch.where(tmp_msk!=0)
            indexes = list(zip(indexes[0].tolist(),indexes[1].tolist()))
            # randomly mask out some syntactic targets
            N_indexes = len(indexes)
            np.random.shuffle(indexes)
            # if "dep" relation is presented in the syntax tree, double the mask percentage.
            if len(loc) > 0:
                perc = self.mask_percentage * 2
            else:
                perc=  self.mask_percentage
            indexes = indexes[:int(N_indexes*(1-perc))]
            selection_indexes_dist.append(indexes)
            ix_to_exclude = list(set(punc_loc).union(set(list(zip(*dep_locs))[0])))
            ixs = [i for i in range(tmp_len) if i not in ix_to_exclude]
            N_ixs = len(ixs)
            np.random.shuffle(ixs)
            ixs = ixs[:int(N_ixs*(1-perc))]
            selection_indexes_depth.append(ixs)
        return selection_indexes_dist, selection_indexes_depth"""
    
    def _get_token_embs(self,wp_embs,maps):
        token_logits = []
        for logit, ma in zip(wp_embs,maps):
            token_logit = self._from_wps_to_token(logit,ma).to(next(self.parameters()).device)
            token_logits.append(token_logit)
        return token_logits

    def _from_wps_to_token(self,wp_embs,span_indexes):
        token_embs = [torch.mean(wp_embs[i1:i2,:],dim=0).unsqueeze(0) for i1,i2 in span_indexes]
        return torch.cat(token_embs,dim=0)
