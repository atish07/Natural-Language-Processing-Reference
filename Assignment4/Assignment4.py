import sys
import gensim
import keras
import numpy as np
import pandas as pd
import string
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Input, Dense, Embedding, Dropout, BatchNormalization, Activation, Bidirectional,Flatten
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
nltk.download('stopwords')
nltk.download('punkt')

if __name__ == "__main__":
    pos_rev = sys.argv[1]
    neg_rev = sys.argv[2]
    neg = [sen.rstrip('\n').translate(str.maketrans('','',string.punctuation)) for sen in open(neg_rev)]
    pos = [sen.rstrip('\n').translate(str.maketrans('','',string.punctuation)) for sen in open(pos_rev)]

    data = pd.DataFrame(columns=['sentence', 'sentiment'])
    data['sentence'] = neg + pos
    data['sentiment'] = [0]*len(neg) + [1]*len(pos)
    data = data.sample(frac=1, random_state=10) # Shuffle the rows
    data.reset_index(inplace=True, drop=True)

    #Tokenizing the sentences
    stop_word = [word for word in stopwords.words('english')]
    word_tokens = [word_tokenize(w.lower()) for w in data['sentence']]
    exception = ["not","no"]
    stop_words = [word for word in stop_word if word not in exception]

    tokens_word = []
    for line in word_tokens:
        tokens = [word for word in line if word not in stop_words]
        tokens_word.append(tokens)
    
    #Creating Word2Vec model
    model_w = gensim.models.Word2Vec(tokens_word, min_count = 10,  
                              size = 300, window = 5)
    model_w.save("word2vec.model")
    w2v = Word2Vec.load("word2vec.model")

    embedding_matrix = w2v.wv.vectors
    embedding_layer = Embedding(input_dim = embedding_matrix.shape[0],
                            output_dim = embedding_matrix.shape[1],
                            weights=[embedding_matrix],trainable=True,input_length=23)

    tokenizer = Tokenizer(num_words = embedding_matrix.shape[0])
    tokenizer.fit_on_texts([' '.join(seq[:23]) for seq in tokens_word])

    X = tokenizer.texts_to_sequences([' '.join(seq[:23]) for seq in tokens_word])
    X = pad_sequences(X, maxlen=23, padding='post', truncating='post')
    y = data['sentiment']

    #Splitting the data into Training,Validation and testing
    X_train_seq, X_test, y_train_seq, y_test = train_test_split(X, y,stratify = y, random_state=42, test_size=0.1)
    X_train, X_val, y_train, y_val = train_test_split(X_train_seq, y_train_seq, random_state=42, test_size=1/9)

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    y_val = np_utils.to_categorical(y_val)

    EMBEDDING_DIM = 300
    BATCH_SIZE = 512
    N_EPOCHS = 10

    #Building a NN
    mod = Sequential()
    mod.add(embedding_layer)
    mod.add(Flatten())
    mod.add(Dense(50, name='dense_1'))
    mod.add(Dropout(rate=0.4, name='dropout_1'))
    mod.add(Activation(activation='tanh', name='activation_1'))
    mod.add(Dense(2, activation='softmax', name='output_layer'))
    mod.summary()

    mod.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    mod.fit(X_train, y_train,
           batch_size=BATCH_SIZE,
          epochs=10,
          validation_data=(X_val, y_val))

    score = mod.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print(score)