pretrained_bert_urls = {"bert":"bert-base-uncased",
                        "biobert":"dmis-lab/biobert-base-cased-v1.1",
                        "scibert":"allenai/scibert_scivocab_uncased",
                        "roberta":"roberta-base"}

class_weights = {"chemprot":[1.4606991761300379, 17.083333333333332, 5.831111111111111, 75.83815028901734, 55.829787234042556, 18.426966292134832],
                 "drugprot":[1.358,45.340,98.397,2232.586,4980.385,66.610,28.814,48.717,46.985,12.023,73.158,70.375,32.324,2697.708]}

class_weights_dist = [1,25.00733896417376,9.516085003293709,6.624811896022926,5.915118085752738,6.259954031527871,7.703170077501162,10.688554262862457,
                      16.02397030648247,25.47309156315425,20.361650810577963,975.9390982768117]

class_weights_depth = [39.4428078001332,7.705854406918652,5.130762055007867,4.398893775500686,5.447390262685035,8.726666483900594,15.822166165469593,
                       30.672777423123712,37.54970656757793,489.9088110811851]

# number of pre-defined distance categories
n_dist_cats = 12
n_depth_cats = 10

# number of pre-defined depth categories

punct_upos_tags = ["PUNCT"]

punct_xpos_tags = ["''", ',', '-LRB-', '-RRB-', '.', ':', '``']
