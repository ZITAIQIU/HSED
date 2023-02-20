# HSED
This repository contains the source code and dataset for the paper "Heterogeneous Social Event Detection via Hyperbolic Graph Representations". This article is under review, only part of the code is put up, if you need all the code, please feel free to contact me.
# Requirements
* Python==3.7
* torch>=1.4.0
* numpy>=1.16.4
* scipy>=1.2.1
* networkx>=2.4
* dgl==0.4.3
* scikit-learn==0.20.3
* torchvision=0.2.2

# Usage
## To run HSED
* Step 1) set model as 'HSED' and dataset as 'twitter' in ```config.py```.
* Step 2) run ```feature_process.py``` to generate the initial features from the Twitter dataset.
* Step 3) run ```generate_homo_graph.py``` to cover social message to homogeneous information network as input.
* Step 4) run ```HSED.py```
## To run UHSED
* Step 1) set model as 'UHSED' and dataset select from ['mini-twitter', 'cora'[2], 'citeseer'[2]] in ```config.pu```.
* Step 2) run ```UHSED.py```




# Datasets
HSED only use Twitter dataset [1] and UHSED use mini-Twitter, Cora [2] and Citeseer [2] datasets.

To run this code on different detasets please change the valu of 'dataset' in ```config.py```.

## Twitter dataset
The Twitter dataset contains 68.841 manually labeled tweets related to 503 event classes.
### Twitter data Format
```
'event_id': manually labeled event class
'tweet_id': tweet id
'text': content of the tweet
'created_at': timestamp of the tweet
'user_id': the id of the sender
'user_loc', 'place_type', 'place_full_name', and 'place_country_code': the location of the sender
'hashtags': hashtags contained in the tweet
'user_mentions': user mentions contained in the tweet
'image_urls': links to the images contained in the tweet
'entities': a list, named entities in the tweet (extracted using spaCy)
'words': a list, tokens of the tweet (hashtags and user mentions are filtered out)
'filtered_words': a list, lower-cased words of the tweet (punctuations, stop words, hashtags, and user mentions are filtered out)
'sampled_words': a list, sampled words of the tweet (only words that are not in the dictionary are kept to reduce the total number of unique words and maintain a sparse message graph)
```
### Twitter dataset Usage
```Python
import pandas as pd
import numpy as np

p_part1 = './data/Twitter_initial/68841_tweets_multiclasses_filtered_0722_part1.npy'
p_part2 = './data/Twitter_initial/68841_tweets_multiclasses_filtered_0722_part2.npy'
df_np_part1 = np.load(p_part1, allow_pickle=True)
df_np_part2 = np.load(p_part2, allow_pickle=True)
df_np = np.concatenate((df_np_part1, df_np_part2), axis = 0)
print("Loaded data.")
df = pd.DataFrame(data=df_np, columns=["event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",\
    "place_type", "place_full_name", "place_country_code", "hashtags", "user_mentions", "image_urls", "entities", 
    "words", "filtered_words", "sampled_words"])
```
# Baselines
In our paper the baselines we used including:
* Word2vec [3]. Source: https://spacy.io/models/en#en_core_web_lg.
* LDA [4]. Source: https://radimrehurek.com/gensim/models/ldamodel.html.
* WMD [5]. Source: https://tedboy.github.io/nlps/generated/generated/gensim.similarities.WmdSimilarity.html#gensim.similarities.WmdSimilarity.
* BERT [6]. Source: https://github.com/huggingface/transformers.
* KPGNN [7]. Source: https://github.com/RingBDStack/KPGNN.
* FinEvent [8]. Source: https://github.com/RingBDStack/FinEvent.
* DGI [9]. Source: https://github.com/PetarV-/DGI.
* GraphCL [10]. Source: https://github.com/Shen-Lab/GraphCL.

# Reference
[1] A. J. McMinn, Y. Moshfeghi, and J. M. Jose, “Building a large-scale corpus for evaluating event detection on twitter,” in Proceedings of the 22nd ACM international conference on Information & Knowledge Management, 2013, pp. 409–418.

[2] P. Sen, G. Namata, M. Bilgic, L. Getoor, B. Galligher, and T. Eliassi-Rad, “Collective classification in network data,” AI magazine, vol. 29, no. 3, pp. 93–93, 2008.

[3] T. Mikolov, K. Chen, G. Corrado, and J. Dean, “Efficient estimation of word representations in vector space,” arXiv preprint arXiv:1301.3781, 2013.

[4] D. M. Blei, A. Y. Ng, and M. I. Jordan, “Latent dirichlet allocation,” Journal of machine Learning research, vol. 3, no. Jan, pp. 993–1022, 2003.

[5] M. Kusner, Y. Sun, N. Kolkin, and K. Weinberger, “From word embeddings to document distances,” in International conference on machine learning. PMLR, 2015, pp. 957–966.

[6] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” arXiv preprint arXiv:1810.04805, 2018.

[7] Y. Cao, H. Peng, J. Wu, Y. Dou, J. Li, and P. S. Yu, “Knowledge-preserving incremental social event detection via heterogeneous gnns,” in Proceedings of the Web Conference 2021, 2021, pp. 3383–3395.

[8] H. Peng, R. Zhang, S. Li, Y. Cao, S. Pan, and P. Yu, “Reinforced, incremental and cross-lingual event detection from social messages,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022.

[9] P. Veliˇckovi ́c, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y. Bengio, “Graph attention networks,” arXiv preprint arXiv:1710.10903, 2017.

[10] Y. You, T. Chen, Y. Sui, T. Chen, Z. Wang, and Y. Shen, “Graph contrastive learning with augmentations,” Advances in Neural Information Processing Systems, vol. 33, pp. 5812–5823, 2020.
