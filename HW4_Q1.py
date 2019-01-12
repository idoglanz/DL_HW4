import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import imdb
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.models import Model
from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, LSTM, Embedding, Dropout, TimeDistributed
from keras.layers.merge import add, concatenate
from keras.callbacks import ModelCheckpoint

# ------------------------------------------------- Load IMDB reviews --------------------------------------------------


def main():
    top_words = 5000    # todo alter for more or less top words
    max_len = 100
    (reviews, sentiment), _= imdb.load_data(num_words=top_words)    # X_train is the word sequence


    x_train, x_test, y_train, y_test = train_test_split(reviews, sentiment,
                                                        test_size=0.5,
                                                        stratify=sentiment)

    [x1_train, x2_train, y_train] = sequence_generator(x_train[:20], y_train[:20], max_len=100)

    [x1_test, x2_test, y_test] = sequence_generator(x_test[:20], y_test[:20], max_len=100)

    y_train = np.array([to_categorical(y, num_classes=top_words) for y in y_train])
    y_test = np.array([to_categorical(y, num_classes=top_words) for y in y_test])

    word_to_id = imdb.get_word_index()
    word_to_id = {k: (v+3) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<OOV>"] = 2
    id_to_word = {v: k for k, v in word_to_id.items()}

    # for i in range(95, 120):
    #     print('{} '.format('+' if x2_train[i] else '-') + ' '.join(id_to_word.get(w) for w in x1_train[i]))
    #     print('{} '.format('+' if x2_train[i] else '-') + ' '.join(id_to_word.get(y_train[i]))) #for w in x2_train[i]))

    print("Train-set size: " + str(len(x1_train)))
    print("Test-set size: " + str(len(x1_test)))
    print(x1_train.shape, x2_train.shape, y_train.shape)

    model = RNN_model(output_size=top_words)
    print("shapes:")
    print(x1_train.shape, x2_train.shape)
    model.fit([x1_train, x2_train], y_train, epochs=20, verbose=2,   # callbacks=[checkpoint] TODO
              validation_data=([x1_test, x2_test], y_test))

    generate_review(model, max_len, id_to_word, 1, 'positive')

    return model, id_to_word

# ------------------------------------------------- Generate sequence -------------------------------------------------


def sequence_generator(reviews, sentiment, max_len=100,):
    x_review = []
    x_sentiment = []
    y = []

    for index, review in enumerate(reviews):
        for i in range(1, len(review)):
            if i <= max_len:
                x_review.append(review[:i])
                y.append(review[i])
                x_sentiment.append(sentiment[index])

    x_review = sequence.pad_sequences(x_review, maxlen=max_len, padding='post', truncating='post')

    return np.array(x_review), np.array(x_sentiment), np.array(y)


def RNN_model(output_size, max_length=100, LSTM_state_size = 512):
    # inputs layer
    review = Input(shape=(max_length,))
    embd = Embedding(5000, 32, mask_zero=True)(review)

    sentiment = Input(shape=(1,))
    fc = Dense(32, activation='relu')(sentiment)

    # merge layer
    merge = add([embd, fc])
    # language model
    # decoder1 = LSTM(LSTM_state_size, return_sequences=True)(merge)
    # decoder2 = LSTM(LSTM_state_size, return_sequences=True)(decoder1)
    # do = Dropout(0.1)(decoder2)
    # decoder3 = LSTM(LSTM_state_size, return_sequences=True)(do)
    # do = Dropout(0.1)(decoder3)

    decoder2 = LSTM(LSTM_state_size)(merge)
    do = Dropout(0.1)(decoder2)

    # fully connected
    decoder4 = Dense(256, activation='relu')(do)
    bn = BatchNormalization()(decoder4)
    do = Dropout(0.1)(bn)
    # decoder5 = Dense(512, activation='relu')(do)
    # bn = BatchNormalization()(decoder5)
    # do = Dropout(0.1)(bn)

    # output
    # outputs = TimeDistributed(Dense(output_size, activation='softmax'))(do)
    outputs = Dense(output_size, activation='softmax')(do)

    model = Model(inputs=[review, sentiment], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    print(model.summary())
    return model


def generate_review(model, max_length, id_to_word, seed, sent='positive'):
    if sent is "positive":
        sentiment = np.array(1)
    else:
        sentiment = np.array(0)
    review = np.zeros(max_length)
    review[0] = seed

    for i in range(2, max_length-1):
        if review[i] != 0:
            next_word = model.predict([review.reshape(-1, max_length, 1), sentiment.reshape(-1, 1, 1)])
            next_word = np.argmax(next_word)
            review[i+1] = next_word
        else:
            break

    print('The predicted word is: ')
    print(' '.join(id_to_word.get(w) for w in review))

    return review


max_len = 100
seed = 1
sent = "positive"

[model, id2word] = main()


