# mts-tacred

Multi-Task-Syntactic-BERT (MTS-BERT) on TACRED dataset.

MODIFICATION:
- no longer entity markers: use [emb_sent,emb_subj,emb_obj]
- not anymore regroup wordpiece embeddings to form token embeddings -> add links between subsquent wordpieces and the head wordpiece

TACRED dataset statistics:
| dataset         |    max # tokens    |   avg # tokens   |  max # wordpieces  | avg # wordpieces |
|-----------------|--------------------|------------------|--------------------|------------------|
|train            |         96         |      37.07       |                    |                  |
|dev              |         95         |      35.46       |                    |                  |
|test             |         96         |      34.75       |                    |                  |
