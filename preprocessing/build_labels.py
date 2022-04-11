# ========= 5 April 2022 ============
# ========= this script build labels for MTS-BERT and other labels, including
#          1) distance between a pair of words in the syntax tree;
#          2) depth for each word in the syntax tree;
#          3) head word index for each word in the syntax tree;
#          4) indexes that correspond to punctuations & special tokens for BERT [CLS], [SEP] (for masking);
# ========= along with tokenization information
#          5) wordpiece ids;
#          6) map from token index to the corresponding wordpiece index range;
#          7) wordpiece ranges that correspond to the subject and object entity;
#          8) relation label;   

import os
import json
import pickle
from argparse import ArgumentParser
from collections import defaultdict
from re import sub
from tqdm import tqdm
from transformers import AutoTokenizer
from constant import punct_upos_tags, punct_xpos_tags

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def get_pair_wise_words(i1,i2,graph):
    if i1 == i2:
        return 0
    
    assert i1 in graph and i2 in graph, "NOT valid word index in the original sentence."
    
    visited = [i1]
    dist = 1
    to_visit = graph[i1]
    
    while len(to_visit) > 0:
        tmp_to_visit = set()
        for i in to_visit:
            if i == i2:
                return dist
            for ii in graph[i]:
                if ii not in visited:
                    tmp_to_visit.add(ii)
        visited += to_visit
        to_visit = list(tmp_to_visit)
        dist += 1
    return -1

def build_dist_matrix(graph,length):
    mat = [[0] * length for _ in range(length)]
    for i in range(length):
        for j in range(i+1,length):
            tmp = get_pair_wise_words(i,j,graph)
            assert tmp != -1, f"ERROR: no distance calculated for {(i,j)} (original word index)."
            mat[i][j] = mat[j][i] = tmp
    return mat

def build_depth_lst(graph,length,root_id):
    # attribute the current depth for each word on the current level during BFS
    res = {root_id:0}
    visited = [root_id]
    dist = 1
    to_visit = graph[root_id]
    
    while len(to_visit) > 0:
        tmp_to_visit = set()
        for i in to_visit:
            res[i] = dist
            for ii in graph[i]:
                if ii not in visited:
                    tmp_to_visit.add(ii)
        visited += to_visit
        to_visit = list(tmp_to_visit)
        dist += 1
    return [res[i] for i in range(length)]

def build_graph(deps):
    # NOTE: add [CLS], [SEP] into the graph
    # use the logic of Stanza: words are 1-indexed
    graph = defaultdict(list)
    for ix, h, _ in deps:
        if h == 0:
            root_ix = ix
        else:
            graph[ix].append(h)
            graph[h].append(ix)
    graph[root_ix].append(0); graph[root_ix].append(len(deps)+1)
    graph[0].append(root_ix); graph[len(deps)+1].append(root_ix)
    return graph, root_ix

def process_sent(tokens,deps,pos,subj_start,subj_end,obj_start,obj_end):
    # map token id -> head wordpiece id
    m = defaultdict(list)
    m[0] = [0,1]
    curr_ix = 1
    wp_ids = [101]
    for i, tk in enumerate(tokens):
        tmp_wps = tokenizer.tokenize(tk)
        wp_ids += tokenizer.convert_tokens_to_ids(tmp_wps) 
        m[i+1].append(curr_ix)
        curr_ix += len(tmp_wps)
        m[i+1].append(curr_ix)
    m[i+2] = [curr_ix,curr_ix+1]
    wp_ids += [102]

    # number of tokens
    N = len(tokens) + 2
    graph, root_ix = build_graph(deps)
    
    # generate: 1) pairwise-distance in syntax trees; 2) depth for each word in syntax trees; 3) dependency graph;
    # extended syntax trees: 1) [CLS], [SEP] linked to the root; 2) subsequent wordpieces linked to the head wordpiece.
    dist_mat = build_dist_matrix(graph,N)
    depth_lst = build_depth_lst(graph,N,root_ix)
    head_ix = [d[1] for d in deps] # [CLS] and [SEP] are excluded

    # mask some tokens (for MTS-BERT): [CLS], [SEP] and punctuations: save only indexes to be masked.
    ky = [0]
    for i, (upos, xpos) in enumerate(pos):
        if upos in punct_upos_tags or xpos in punct_xpos_tags:
            ky.append(i+1)
    ky.append(N-1)

    # wordpiece index range corresponding to the subject and the object
    subj_wp_range = (m[subj_start+1][0],m[subj_end+1][-1])
    obj_wp_range = (m[obj_start+1][0],m[obj_end+1][-1]) 

    return wp_ids, m, dist_mat, depth_lst, head_ix, ky, subj_wp_range, obj_wp_range

def recover_subj_or_obj(wps):
    res = []
    tmp_token = ""
    for wp in wps:
        if not wp.startswith("##"):
            if len(tmp_token) > 0:
                res.append(tmp_token)
            tmp_token = wp
        else:
            tmp_token += wp.strip("##")
    res.append(tmp_token)
    return res

def check_legitimacy(doc,wp_ids,m,subj_wp_range,obj_wp_range):
    tokens = doc["token"]
    subj_token = tokens[doc["subj_start"]:doc["subj_end"]+1]
    obj_token = tokens[doc["obj_start"]:doc["obj_end"]+1]
    wps = tokenizer.convert_ids_to_tokens(wp_ids)

    # verify that [UNK] not presented in the tokenization result
    if 100 in wp_ids:
        print("[UNK] detected.")

    # remap wordpieces to tokens
    re_tokens = []
    for i1, i2 in m.values():
        tmp_token = []
        for s in wps[i1:i2]:
            if s.startswith("##"):
                tmp_token.append(s[2:])
            else:
                tmp_token.append(s)
        re_tokens.append(''.join(tmp_token))
    subj_wps = wps[subj_wp_range[0]:subj_wp_range[1]]
    obj_wps = wps[obj_wp_range[0]:obj_wp_range[1]]

    # recover subject and object tokens
    re_subj_token = recover_subj_or_obj(subj_wps)
    re_obj_token = recover_subj_or_obj(obj_wps)

    return 100 not in wp_ids and \
           (re_tokens[1:-1] == [t.lower() for t in tokens]) and \
           (''.join(re_subj_token) == ''.join([t.lower() for t in subj_token])) and \
           (''.join(re_obj_token) == ''.join([t.lower() for t in obj_token]))

def process(args,dataset_fn,rel2id):
    data = json.load(open(os.path.join(args.data_dir,f"{dataset_fn}.json"),'r'))
    processed_data = []
    error_indexes = []
    for doc_id, doc in tqdm(enumerate(data),desc=f"processing {dataset_fn} set..."):
        wp_ids, m, dist_mat, depth_lst, head_ix, key, subj_range, obj_range = process_sent(doc["token"],doc["stanza_dep"],doc["stanza_pos"],
                                                                                           doc["subj_start"],doc["subj_end"],
                                                                                           doc["obj_start"],doc["obj_end"])
        new_doc = {"wp_ids":wp_ids,"map":m,"distance_matrix":dist_mat,"depth_list":depth_lst,"head_indexes":head_ix,
                   "mask_indexes":key,"subj_range":subj_range,"obj_range":obj_range,"label":rel2id[doc["relation"]]}
        if not check_legitimacy(doc,wp_ids,m,subj_range,obj_range):
            error_indexes.append(doc_id)
        processed_data.append(new_doc)

    with open(os.path.join(args.output_dir,f"{dataset_fn}.pkl"),"wb") as f:
        pickle.dump(processed_data,f,pickle.HIGHEST_PROTOCOL)

    if len(error_indexes) == 0:
        print(f"succeed on {dataset_fn}.")
    else:
        print(f"ERROR on {dataset_fn}: {len(error_indexes)} errors.")
        with open(os.path.join(args.output_dir,f"{dataset_fn}_error_indexes.txt"),'w') as f:
            f.write('\n'.join([str(i) for i in error_indexes]))

def main(args):
    rel2id = pickle.load(open(os.path.join(args.data_dir,"rel2id.pkl"),"rb"))
    process(args,"train",rel2id)
    process(args,"dev",rel2id)
    process(args,"test",rel2id)

if __name__ == "__main__":
    parser = ArgumentParser(description='Build MTS labels of TACRED.')
    parser.add_argument("--data_dir",type=str,help="path to input data.")
    parser.add_argument("--output_dir",type=str,help="output path to store generated files.")
    args = parser.parse_args()
    main(args)