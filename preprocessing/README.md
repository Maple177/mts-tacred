processing of data for MTS-BERT

##### generate labels 
- pairwise distance between words in syntax trees; (MTS-BERT)
- depth of each word in syntax trees; (MTS-BERT)
- index of the syntactic head word for each word; (MTS-BERT)
- dependency graph; (Late-Fusion)  

##### input & output
- input: pickled list of dictionaries with keys  
  token | relation | subj_start | subj_end | obj_start | obj_end | stanza_pos | stanza_dep
  (subj = token[subj_start:subj_end+1])
  (see example of a demo input file)
- output: pickled dictionary with keys
  wp_id | relation | pairwise_distance | depth | head | map_wp2token | ix_mask
  
##### usage
