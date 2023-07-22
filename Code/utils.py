from typing import List
import pandas as pd
import numpy as np
import re
import nltk



def read_dataset(path):
    csv = pd.read_csv(path)
    text= csv["text"]
    category= csv["category"]
    stance= csv["stance"]

    return text , category , stance

def read_testset(path):
    csv = pd.read_csv(path)
    ids= csv["id"]
    text= csv["text"]

    return ids,text

def write_test_file(ids,categories,stances,path="Output/5.csv"):

    df= pd.DataFrame({
        "id":ids,
        "category":categories,
        "stance": stances          
    })
    print(df.columns)
    df.to_csv(path,index= False)
    

def build_vocab(tokenized_sentences):
    vocab = set()
    for sentence in tokenized_sentences:
        vocab=vocab.union(set(sentence))
    return vocab

def combine_pipe(methods:List):
    """
        combine multiple functions into one 
    """
    def combined(txt):
        res = txt 
        for method in methods:
            res = method(res)
        return res
    
    return combined


def remove_with_regex(regex,replace=""):
    def remover(txt):
        return re.sub(regex,replace,txt)
    return remover

remove_urls = remove_with_regex(f"http\S*")
remove_lfs = remove_with_regex(f"<LF>")
remove_user_tag = remove_with_regex(f"@USER")
remove_under_scores = remove_with_regex(f"_"," ")

def remove_words_in(vocab,to_remove):

    new_vocab = set()
    for v in vocab:
        if v not in to_remove:
            new_vocab.add(v)
    return new_vocab

# def input_tweets_to_features(tweets, extract_features):
#     """
#     This function takes the tweets as strings and extracts the features for every tweet
    
#     Input: 
#     - tweets: list of strings (tweetdev_Xweets), 2) 
#     """
#     features_count= len(extract_features(tweets[0]).flatten().tolist())
#     X = np.zeros((len(tweets), features_count))
    
#     for i, tweet in enumerate(tweets):
#         X[i] = extract_features(tweet)
#     ###################################################################################################################
    
#     return X

from matplotlib import pyplot as plt

def draw_features(X,Y,C):
    plt.scatter(X,Y, marker=".", color= ["red" if c else "blue" for c in C])
    plt.show()
