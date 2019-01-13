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
from random import randint
from nltk.translate.bleu_score import corpus_bleu

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
        n_sequences = randint(50, 100)
        for i in range(1, len(review)):
            if i <= n_sequences:
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


def generate_review(model, max_length, id_to_word, seed, sent='positive', mix_ind=None):
    if sent is "positive":
        if mix_ind is None:
            sentiment = np.ones((1, max_length))
        else:
            sentiment = np.ones((1, max_length))
            sentiment[0, mix_ind:] = 0

    elif sent is "negative":
        if mix_ind is None:
            sentiment = np.zeros((1, max_length))

        else:
            sentiment = np.zeros((1, max_length))
            sentiment[0, mix_ind:] = 1

    review = np.zeros((1, max_length))
    print(sentiment)

    for i in range(len(seed)):
        review[0, i] = seed[i]

    for i in range(len(seed) - 1, max_length - 1):
        if review[0, i] != 0.0:
            next_word = model.predict([review, sentiment[:, i]])
            next_word = soft_sample(next_word, greedy=False, ignore_OOV=True)
            review[0, i + 1] = next_word
        else:
            break

    rev4print = review[0, :].tolist()

    print('The predicted word is: ')
    print(' '.join(id_to_word.get(w) for w in rev4print))

    return review


def soft_sample(options, greedy=False, ignore_OOV=True, n_max=1):
    if greedy is True:
        if ignore_OOV is False:
            return np.argmax(options)
        else:
            while np.argmax(options) == 2:
                options[0, int(np.argmax(options))] = 0
            #             return np.argmax(options)
            return n_max(options, n_max, rand=True)
    else:
        norm_preds = (options / (np.sum(options + 0.0000000001)))[0, :].tolist()
        chosen = np.random.multinomial(1, norm_preds, 1)
        if ignore_OOV is False:
            return np.argmax(chosen)
        else:
            while np.argmax(chosen) == 2:
                options[0, int(np.argmax(chosen))] = 0
                norm_preds = (options / (np.sum(options + 0.0000000001)))[0, :].tolist()
                chosen = np.random.multinomial(1, norm_preds, 1)
    return np.argmax(chosen)


def n_max(array, n, rand=True):
    if rand is True:
        n = randint(1, n)
    for i in range(n - 1):
        array[0, np.argmax(array)] = 0
    return np.argmax(array)


def seed_gen(sentence, word2id=None):
    sentence = ' '.join(("<START>", sentence))
    if word2id is None:
        word2id = imdb.get_word_index()
        word2id = {k: (v + 3) for k, v in word2id.items()}
        word2id["<PAD>"] = 0
        word2id["<START>"] = 1
        word2id["<OOV>"] = 2

    seed_out = [word2id[word] for word in sentence.split()]
    return seed_out


def BLEU_evaluate(sampled_data, prediction):
    score = [0]*4
    for i in range(len(sampled_data)):
        score[0] += corpus_bleu(sampled_data[i,:], prediction, weights=(1.0, 0, 0, 0))
        score[1] += corpus_bleu(sampled_data[i,:], prediction, weights=(0.5, 0.5, 0, 0))
        score[2] += corpus_bleu(sampled_data[i,:], prediction, weights=(0.3, 0.3, 0.3, 0))
        score[3] += corpus_bleu(sampled_data[i,:], prediction, weights=(0.25, 0.25, 0.25, 0.25))
    score /= len(sampled_data)
    return score


seed_str = 'the movie was'
generate_review(model, max_len, id2word, seed=seed_gen(seed_str, word2id=word2id), sent="positive", mix_ind=10)

max_len = 100
seed = 1
sent = "positive"

# [model, id2word] = main()
seed_sentence = 'the movie was'
seed_num = seed_gen(seed_sentence)
print(seed_num)
