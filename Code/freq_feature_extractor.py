import numpy as np
import pandas as pd

def _build_frequency(tweets, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in tweet:
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs




def build_feature_extractor(tweets, ys):
    
    freqs = _build_frequency(tweets, ys)

    def extract_features(tweets:pd.Series):
        '''
        Input: 
            tweet: tokens of the tweet
        Output: 
            x: a feature vector of dimension (1,3)
        '''
        
        X = np.zeros((tweets.shape[0],3))
        # 3 elements in the form of a 1 x 3 vector
        # x = np.zeros((1, 3)) 
        
        ############################## TODO: Calculate positive and negative features ##################################
        # loop through each word in the list of words
        for i,tweet in enumerate(tweets):

            for word in tweet:
                if (word, 1.0) in freqs:
                    X[i, 2] += freqs[(word, 1.0)]
                if (word, 0.0) in freqs:
                    X[i, 1] += freqs[(word, 0.0)] 
                if (word, -1.0) in freqs:
                    X[i, 0] += freqs[(word, -1.0)] 
        return X

    return extract_features