from __future__ import absolute_import, division, print_function

import logging
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pickle

import torch
import transformers 
from transformers import BertConfig
from sklearn.metrics import f1_score

from opt import get_args
from loader import DataLoader
from model import (SyntaxBertModel, set_seed)
from utils.constant import pretrained_bert_urls

transformers.logging.set_verbosity_error()
logger = logging.getLogger(__name__)

def oh(v):
    oh_vectors = np.zeros_like(v)
    for i, line in enumerate(v):
        oh_vectors[i,np.argmax(line)] = 1
    return oh_vectors

def evaluate(dataloader,model,predict_only=False):
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    all_preds = []
    all_golds = []

    for batch in dataloader:
        with torch.no_grad():
            if predict_only:
                logits = model(**batch,predict_only=predict_only)[0]
            else:
                all_golds.append(batch["labels"].detach().cpu().numpy())
                loss, logits = model(**batch,predict_only=predict_only)[:2]
            
            all_preds.append(oh(logits.detach().cpu().numpy()))

            if not predict_only:
                eval_loss += loss.item()
                nb_eval_steps += 1
    
    all_preds = np.concatenate(all_preds)
    if not predict_only:
        all_golds = np.concatenate(all_golds)
        eval_score = f1_score(all_golds,all_preds,average="micro")
        eval_loss = eval_loss / nb_eval_steps
        return eval_loss, eval_score
    else:
        return all_preds

def main():
    start_time = time.time()
    args = get_args()
    
    # Setup CUDA, GPU & distributed training
    if not args.force_cpu and not torch.cuda.is_available():
        logger.info("NO available GPU. STOPPED. If you want to continue without GPU, add --force_cpu")
        return 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    log_fn = f"logging/inference_log_{args.run_id}"
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%m/%d/%Y %H:%S',level=logging.INFO,filename=log_fn,filemode='w')
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu)

    if not args.config_name_or_path:
        config_file_name = f"syntax-enhanced-RE/config/{args.model_type}.json"
        assert os.path.exists(config_file_name), "requested BERT model variant not in the preset. You can place the corresponding config file under the folder /config/"
        args.config_name_or_path = config_file_name

    config = BertConfig.from_pretrained(args.config_name_or_path)

    if args.dev:
        eval_dataloader = DataLoader(args.data_dir,"dev",args.mode,args.seed,args.batch_size,args.device,args.debug)
    else:
        eval_dataloader = DataLoader(args.data_dir,"test",args.mode,args.seed,args.batch_size,args.device,args.debug)

    OUTPUT_DIR = os.path.join(args.model_dir,f"lr_{args.learning_rate}_alpha_{args.alpha}/","preds")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    all_preds = []
    for ensemble_id in range(1,args.ensemble_size+1):
        set_seed(args,ensemble_id)
        model_dir = os.path.join(args.model_dir,f"lr_{args.learning_rate}_alpha_{args.alpha}/ensemble_{ensemble_id}")   
        model = SyntaxBertModel.from_pretrained(model_dir,config=config,dataset_name=args.dataset_name,num_labels=args.num_labels,
                                                mode=args.mode,coef=args.alpha)
        model.to(args.device)
        tmp_preds = evaluate(eval_dataloader,model,True)
        all_preds.append([np.argmax(tmp_preds,1)])
    all_preds = np.concatenate(all_preds)

    if args.dev:
        np.save(open(os.path.join(OUTPUT_DIR,"dev_preds.npy"),"wb"),all_preds)
    else:
        np.save(open(os.path.join(OUTPUT_DIR,"test_preds.npy"),"wb"),all_preds)
    
    end_time = time.time()
    logger.info(f"time consumed (inference): {(end_time-start_time):.3f} s.")
    logger.info("probe score on the test set saved.")
    
if __name__ == "__main__":
    main()
