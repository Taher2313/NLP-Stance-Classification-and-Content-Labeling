import gensim 
from gensim.models.keyedvectors import KeyedVectors
  
embeddings = KeyedVectors.load_word2vec_format('fasttextD300W1.bin', binary=True)
