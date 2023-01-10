# HSED
This repository contains the source code and dataset for the paper "Heterogeneous Social Event Detection via Hyperbolic Graph Representations"
# Requirements
* torch>=1.4.0
* numpy>=1.16.4
* scipy>=1.2.1
* networkx>=2.4
* dgl==0.4.3
* scikit-learn==0.20.3
* torchvision=0.2.2
# Datasets and Usage
## Twitter dataset
The Twitter dataset [1] contains 68.841 manually labeled tweets related to 503 event classes. To reduce data processing time, already processed data can be found on [Google Drive](https://drive.google.com/drive/folders/1mb8IT7uTW-gCnK5EFE67iFk7RtZTz3rB?usp=sharing)
## Twitter data format
'''
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
'''


# Reference
[1] Andrew J McMinn, Yashar Moshfeghi, and Joemon M Jose. 2013. Building a large-scale corpus for evaluating event detection on twitter. In Proceedings of the CIKM.ACM, 409–418.
