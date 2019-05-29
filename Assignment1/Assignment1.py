import sys
import nltk
from nltk.corpus import stopwords
import string
import numpy as np

if __name__ == "__main__":
    input_path = sys.argv[1]
    pattern = '!"#$%&()*+,/:;<=>?@[\\]^_`{|}~\t\n'
    sentences = [sen.rstrip('\n').translate(str.maketrans('','',pattern)).replace('.',' . ').replace('-',' - ').replace('\'',' \' ') for sen in open(input_path)]
    stop_words = [word for word in stopwords.words('english')]
    tokens = [word.lower().split() for word in sentences]

    #removing stop words
    tokens_no_stopwords = []
    for line in tokens:
        token = [word for word in line if word not in stop_words]
        tokens_no_stopwords.append(token)

    #Creating Test, Validate and Train sets
    train_list,val_list,test_list = np.split(tokens,[int(0.8*len(tokens)),int(.9*len(tokens))])

    train_list_no_stopword,val_list_no_stopword,test_list_no_stopword = np.split(tokens_no_stopwords,[int(0.8*len(tokens_no_stopwords)),int(.9*len(tokens_no_stopwords))])
    
    """
    Tokenize the input file here
    Create train, val, and test sets
    """

    # sample_tokenized_list = [["Hello", "World", "."], ["Good", "bye"]]

    np.savetxt("train.csv", train_list, delimiter=",", fmt='%s')
    np.savetxt("val.csv", val_list, delimiter=",", fmt='%s')
    np.savetxt("test.csv", test_list, delimiter=",", fmt='%s')

    np.savetxt("train_no_stopword.csv", train_list_no_stopword,
               delimiter=",", fmt='%s')
    np.savetxt("val_no_stopword.csv", val_list_no_stopword,
               delimiter=",", fmt='%s')
    np.savetxt("test_no_stopword.csv", test_list_no_stopword,
               delimiter=",", fmt='%s')
