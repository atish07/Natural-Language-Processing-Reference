import numpy as np
import sys
import string
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

if __name__ == "__main__":
    neg_sen = [sen.rstrip('\n').translate(str.maketrans('','',string.punctuation)) for sen in open("neg.txt")]
    pos_sen = [sen.rstrip('\n').translate(str.maketrans('','',string.punctuation)) for sen in open("pos.txt")]
    sen = neg_sen + pos_sen

    stop_words = [word for word in stopwords.words('english')]
    word_tokens = [word_tokenize(w.lower()) for w in sen]

    tokens_wo_stopwords = []
    for line in word_tokens:
        tokens = [word for word in line if word not in stop_words]
        tokens_wo_stopwords.append(tokens)

    model = Word2Vec(tokens_wo_stopwords, min_count = 10,  
                              size = 100, window = 5)
    
    model.save("word2vec.model")
    model = Word2Vec.load("word2vec.model")

    good_similar = model.most_similar(["good"],topn=20)
    bad_similar = model.most_similar(["bad"],topn=20)

    print("20 words Similar to good are:",good_similar,"\n")
    print("20 words similar to bad are:",bad_similar)