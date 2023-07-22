
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def _preprocess_for_TfidfVectorizer(corpus_tokenized:pd.Series):
    """
    This function takes list of documents and preprocess them for CountVectorizer
    
    Inputs:
    - corpus: List of strings ( tweets )
    
    Returns:
    - processed_corpus: List of strings
    """

    return corpus_tokenized.apply(lambda x : " ".join(x))


def _train_tfidf_vectorizer(processed_train_corpus):
    """
    This function takes processed training corpus and trains a CountVectorizer
    
    Inputs:
    - processed_train_corpus: list of tweets
    
    Returns:
    - vectorizer: TfidfVectorizer Object after training
    """
    # print(processed_train_corpus)
    vectorizer = None
    # Create the Vectorizer
    vectorizer = TfidfVectorizer(token_pattern=r"\S+")
    
    # fit the vectorizer
    vectorizer.fit(processed_train_corpus)
    ##########################################################################################################
    
    return vectorizer

def build_feature_extractor(corpus_tokenized:pd.Series):
    preprocessed_text_for_countvectorizer = _preprocess_for_TfidfVectorizer(corpus_tokenized)
    vectorizer = _train_tfidf_vectorizer(preprocessed_text_for_countvectorizer)
    
    def extract_features(preprocessed_tweets):
        '''
        Input: 
            tweet: tokens of the tweet
        Output: 
            x: a feature vector of dimension (1,V)
        '''
        

        return vectorizer.transform(preprocessed_tweets.apply(lambda x : " ".join(x)))
    
    return extract_features

from sklearn.feature_extraction.text import TfidfVectorizer

# def extract_tfidf(train,dev):
#     # body_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='arabic',token_pattern=r"\S+")#, max_features=1024)
#     vectorizer = TfidfVectorizer( token_pattern=r"\S+")#, max_features=1024)
#     text_tfidf = vectorizer.fit_transform(train)

#     # Tranform dev/test bodies and headlines using the trained vectorizer (trained on training data)
#     dev_tfidf = vectorizer.transform(dev)

  
    
#     feature_names = np.array(vectorizer.get_feature_names())
#     sorted_by_idf = np.argsort(vectorizer.idf_) 
#     print('Features with lowest and highest idf in the body vector:\n')
#     # The token which appears maximum times but it is also in all documents, has its idf the lowest
#     print("Features with lowest idf:\n{}".format(
#     feature_names[sorted_by_idf[:10]]))
#     # The tokens can have the most idf weight because they are the only tokens that appear in one document only
#     print("\nFeatures with highest idf:\n{}".format(
#     feature_names[sorted_by_idf[-10:]]))



#     return text_tfidf, dev_tfidf