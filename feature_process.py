import os.path
import numpy as np
import pandas as pd
import spacy
from datetime import datetime
import random
import nlpaug.augmenter.word as naw
from nltk import word_tokenize
from nltk.corpus import stopwords


# Change other entities in message into feature vector
def documents_to_features(df):
    nlp = spacy.load("en_core_web_lg")
    features = df.filtered_words.apply(lambda x: nlp(' '.join(x)).vector).values
    return np.stack(features, axis=0)


# encode one times-tamp
# t_str: a string of format '2012-10-11 07:19:34'
def extract_time_feature(t_str):
    t = datetime.fromisoformat(str(t_str))
    OLE_TIME_ZERO = datetime(1899, 12, 30)
    delta = t - OLE_TIME_ZERO
    return [(float(delta.days) / 100000.), (float(delta.seconds) / 86400)]  # 86,400 seconds in day

# encode the times-tamps of all the messages in the dataframe
def df_to_t_features(df):
    t_features = np.asarray([extract_time_feature(t_str) for t_str in df['created_at']])
    return t_features



# path
load_path = './data/Twitter_initial/'
save_path = './data/'

#load twitter dataset
p_part1 = load_path + '68841_tweets_multiclasses_filtered_0722_part1.npy'
p_part2 = load_path + '68841_tweets_multiclasses_filtered_0722_part2.npy'
df_np_part1 = np.load(p_part1, allow_pickle=True)
df_np_part2 = np.load(p_part2, allow_pickle=True)
df_np = np.concatenate((df_np_part1, df_np_part2), axis=0)
print("Loaded data.")
df = pd.DataFrame(data=df_np, columns=["event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",\
    "place_type", "place_full_name", "place_country_code", "hashtags", "user_mentions", "image_urls", "entities",
    "words", "filtered_words", "sampled_words"])

print("Data converted to dataframe.")
print('Data shape:', df.shape)
print('======================Data processing======================')


processed_save_path = save_path + 'twitter/'
if not os.path.exists(processed_save_path):
    os.mkdir(processed_save_path)

offline_save_path = processed_save_path + 'offline/'
if not os.path.exists(offline_save_path):
    os.mkdir(offline_save_path)

offline_save_block = offline_save_path + '0/'
if not os.path.exists(offline_save_block):
    os.mkdir(offline_save_block)


#original feature

d_features = documents_to_features(df)
print("Document original features generated.")

t_features = df_to_t_features(df)
print("Time features generated.")

combined_features = np.concatenate((d_features, t_features), axis=1)
print("Concatenated document features and time features.")

np.save(offline_save_block + 'features.npy', combined_features)
print("Offline original features saved.")

labels = [int(each) for each in df['event_id'].values]
np.save(offline_save_block + 'labels.npy', labels)
print("Offline labels saved.")
print('Total features shape', combined_features.shape)