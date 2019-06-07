import sys
import nltk
from nltk.corpus import stopwords
import string
import numpy as np
from nltk import ngrams
from nltk import everygrams
import sys
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier

if __name__ == "__main__":
    train_pos = sys.argv[1]
    train_neg = sys.argv[2]
    val_pos = sys.argv[3]
    val_neg = sys.argv[4]
    test_pos = sys.argv[5]
    test_neg = sys.argv[6]

    #reading the files
    list_train_neg = []
    list_train_pos = []

    for line in open(train_neg, "r"):
      list_train_neg.append(eval(line))
    for line in open(train_pos, "r"):
      list_train_pos.append(eval(line))

    list_val_neg = []
    list_val_pos = []

    for line in open(val_neg, "r"):
      list_val_neg.append(eval(line))
    for line in open(val_pos, "r"):
      list_val_pos.append(eval(line))

    list_test_neg =[]
    list_test_pos =[]

    for line in open(test_neg, "r"):
      list_test_neg.append(eval(line))
    for line in open(test_pos, "r"):
      list_test_pos.append(eval(line))


    #creating n grams
    def create_ngram_features(words, n=2,both = False):
        if both == False:
            ngram_vocab = ngrams(words, n)
            my_dict = dict([(ng, True) for ng in ngram_vocab])
            return my_dict
        else:
            ngram_vocab = everygrams(words,1, n)
            my_dict = dict([(ng, True) for ng in ngram_vocab])
            return my_dict

    #creating classifier
    def create_classifier(list_train_neg,list_train_pos,list_val_neg,list_val_pos,list_test_neg,list_test_pos,n,grams):
        neg_train_data = []
        for line in list_train_neg:
            words = line
            neg_train_data.append((create_ngram_features(words,n,grams),"negative"))
    
        pos_train_data = []
        for line in list_train_pos:
            words = line
            pos_train_data.append((create_ngram_features(words,n,grams),"positive"))
    
    
        neg_val_data = []
        for line in list_val_neg:
            words = line
            neg_val_data.append((create_ngram_features(words,n,grams),"negative"))
    
        pos_val_data = []
        for line in list_val_pos:
            words = line
            pos_val_data.append((create_ngram_features(words,n,grams),"positive"))
    
    
        neg_test_data = []
        for line in list_test_neg:
            words = line
            neg_test_data.append((create_ngram_features(words,n,grams),"negative"))
    
        pos_test_data = []
        for line in list_test_pos:
            words = line
            pos_test_data.append((create_ngram_features(words,n,grams),"positive"))
        
    
        train_set = neg_train_data + pos_train_data
        val_set = neg_val_data + pos_val_data
        test_set = neg_test_data + pos_test_data


        classifier = NaiveBayesClassifier.train(train_set)
        accuracy_val = nltk.classify.util.accuracy(classifier, val_set)
        accuracy = nltk.classify.util.accuracy(classifier, test_set)
        if(n == 1 and grams == False):
            print('Unigram accuracy for Validation set:', accuracy_val)
            print('Unigram accuracy for Test set:', accuracy)

        elif(n == 2 and grams == False):
            print('Bigram accuracy for Validation set:', accuracy_val)
            print('Bigram accuracy for Test set:', accuracy)

        else:
            print('Unigram + Bigram accuracy for Validation set:', accuracy_val)
            print('Unigram + Bigram accuracy for Test set:', accuracy)

    create_classifier(list_train_neg,list_train_pos,list_val_neg,list_val_pos,list_test_neg,list_test_pos,1,False)

    create_classifier(list_train_neg,list_train_pos,list_val_neg,list_val_pos,list_test_neg,list_test_pos,2,False)

    create_classifier(list_train_neg,list_train_pos,list_val_neg,list_val_pos,list_test_neg,list_test_pos,2,True)


