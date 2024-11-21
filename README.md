# Anc2vec

This repository is a changed version of the original [anc2vec](https://github.com/aedera/anc2vec) repository. The change is to use SBERT embeddings instead of one-hot encoding as the input to the model. SBERT embeddings were generated using a pretrained model (based on BioBERT), the exact generation technique can be inspected [here](https://github.com/mmtftr/OWL2Vec-Star/blob/194c4330e819cfb740b462a4e6b5a049d5e311a6/mmtf-eval/biobert.ipynb)

The code to train the model in this manner can be found in the `test-train.py` file.

We used the 2024-09-28 version of the ontology.