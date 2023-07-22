
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def _preprocess_for_CountVectorizer(corpus_tokenized:pd.Series):
    """
    This function takes list of documents and preprocess them for CountVectorizer
    
    Inputs:
    - corpus: List of strings ( tweets )
    
    Returns:
    - processed_corpus: List of strings
    """

    return corpus_tokenized.apply(lambda x : " ".join(x))


def _train_count_vectorizer(processed_train_corpus):
    """
    This function takes processed training corpus and trains a CountVectorizer
    
    Inputs:
    - processed_train_corpus: list of tweets
    
    Returns:
    - vectorizer: CountVectorizer Object after training
    """
    # print(processed_train_corpus)
    vectorizer = None
    # Create the Vectorizer
    # hint1: check CountVectorizer from sklearn
    # hint2: You will need to specify the token_pattern parameter as the default one will miss some tokens
    vectorizer = CountVectorizer(token_pattern=r"\S+")
    
    # fit the vectorizer
    vectorizer.fit(processed_train_corpus)
    ##########################################################################################################
    
    return vectorizer

def build_feature_extractor(corpus_tokenized:pd.Series):
    preprocessed_text_for_countvectorizer = _preprocess_for_CountVectorizer(corpus_tokenized)
    vectorizer = _train_count_vectorizer(preprocessed_text_for_countvectorizer)
    
    def extract_features(preprocessed_tweets):
        '''
        Input: 
            tweet: tokens of the tweet
        Output: 
            x: a feature vector of dimension (1,V)
        '''
        

        return vectorizer.transform(preprocessed_tweets.apply(lambda x : " ".join(x)))
    
    return extract_features