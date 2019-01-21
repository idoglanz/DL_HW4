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
from keras.models import load_model

# ------------------------------------------------- Load IMDB reviews --------------------------------------------------


def main():
    top_words = 6000  # TODO alter for more or less top words
    max_len = 40
    n_samples = 9000
    (reviews, sentiment), _ = imdb.load_data(num_words=top_words)  # X_train is the word sequence

    x_train, x_test, y_train, y_test = train_test_split(reviews, sentiment,
                                                        test_size=0.5,
                                                        stratify=sentiment)

    [x1_train, x2_train, y_train] = sequence_generator(x_train[:n_samples], y_train[:n_samples], max_len=max_len)

    [x1_test, x2_test, y_test] = sequence_generator(x_test[:30], y_test[:30], max_len=max_len)

    y_train = np.array([to_categorical(y, num_classes=top_words) for y in y_train])
    y_test = np.array([to_categorical(y, num_classes=top_words) for y in y_test])

    word_to_id = imdb.get_word_index()
    word_to_id = {k: (v + 3) for k, v in word_to_id.items()}
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

    model = RNN_model(output_size=top_words, max_length=max_len)

    filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    model.fit([x1_train, x2_train], y_train, epochs=10, verbose=1, batch_size=128, callbacks=[checkpoint],
              validation_data=([x1_test, x2_test], y_test))

    return model, id_to_word, word_to_id


# ------------------------------------------------- Generate sequence -------------------------------------------------


def sequence_generator(reviews, sentiment, max_len=40, ):
    x_review = []
    x_sentiment = []
    y = []

    for index, review in enumerate(reviews):
        n_sequences = randint(10, max_len)
        for i in range(1, len(review)):
            if i <= n_sequences:
                x_review.append(review[:i])
                y.append(review[i])
                x_sentiment.append(sentiment[index])

    x_review = sequence.pad_sequences(x_review, maxlen=max_len, padding='pre', truncating='post')

    return np.array(x_review), np.array(x_sentiment), np.array(y)


def RNN_model(output_size, max_length=40, LSTM_state_size=256):
    # inputs layer:
    # input review sentence and embed
    review = Input(shape=(max_length,))
    embd = Embedding(8000, 32, mask_zero=True)(review)

    # input sentiment for next word generation (0 for negative, 1 for positive)
    sentiment = Input(shape=(1,))
    fc = Dense(32, activation='relu')(sentiment)

    # merge input layers
    merge = add([embd, fc])

    # LSTM zone:
    decoder1 = LSTM(LSTM_state_size, return_sequences=True)(merge)
    decoder2 = LSTM(LSTM_state_size, return_sequences=True)(decoder1)
    do = Dropout(0.5)(decoder2)
    decoder3 = LSTM(LSTM_state_size)(do)
    do = Dropout(0.5)(decoder3)

    # Fully connected part:
    decoder4 = Dense(256, activation='sigmoid')(do)
    bn = BatchNormalization()(decoder4)
    do = Dropout(0.2)(bn)
    decoder5 = Dense(512, activation='relu')(do)
    bn = BatchNormalization()(decoder5)
    do = Dropout(0.2)(bn)

    # output:
    outputs = Dense(output_size, activation='softmax')(do)

    model = Model(inputs=[review, sentiment], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # summarize model
    print(model.summary())
    return model


[model, id2word, word2id] = main()

from nltk.translate.bleu_score import corpus_bleu
from keras.models import load_model


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
            next_word = soft_sample(next_word, greedy=False, ignore_OOV=True, n_max=3)
            review[0, i + 1] = next_word
        else:
            break

    rev4print = review[0, :].tolist()

    print('The predicted word is: ')
    print(' '.join(id_to_word.get(w) for w in rev4print))

    return 1


def soft_sample(options, greedy=True, ignore_OOV=True, n_max=3):
    n_max_rand = np.random.randint(1, n_max)
    if greedy is True:
        if ignore_OOV is False:
            return nth_argmax(options, n_max_rand, rand=True)
        else:
            while np.argmax(options) == 2:
                options[0, int(np.argmax(options))] = 0
            return np.argmax(options)
    #             return nth_argmax(options, n_max, rand=True)
    else:
        norm_preds = (options / (np.sum(options + 0.0000000001)))[0, :].tolist()
        chosen = np.random.multinomial(1, norm_preds, 1)
        if ignore_OOV is False:
            return np.argmax(chosen)
        else:
            while np.argmax(chosen) == 2:
                options[0, int(np.argmax(chosen))] = 0
                norm_preds = (options / (np.sum(options + 0.0000000001)))[0, :].tolist()
                chosen = np.random.multinomial(3, norm_preds, 1)
    return np.argmax(chosen)


def nth_argmax(array, n, rand=True):
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
    score = [0] * 4
    for i in range(len(sampled_data)):
        score[0] += corpus_bleu(sampled_data[i, :], prediction, weights=(1.0, 0, 0, 0))
        score[1] += corpus_bleu(sampled_data[i, :], prediction, weights=(0.5, 0.5, 0, 0))
        score[2] += corpus_bleu(sampled_data[i, :], prediction, weights=(0.3, 0.3, 0.3, 0))
        score[3] += corpus_bleu(sampled_data[i, :], prediction, weights=(0.25, 0.25, 0.25, 0.25))
    score /= len(sampled_data)
    return score


def gen_dicitonary():
    word_to_id = imdb.get_word_index()
    word_to_id = {k: (v + 3) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<OOV>"] = 2
    id_to_word = {v: k for k, v in word_to_id.items()}

    return word_to_id, id_to_word


# load the model
# model = load_model('model-ep005-loss4.666-val_loss5.039.h5')

# re-generate the encryption
[word2id, id2word] = gen_dicitonary()

# user seed input
seed_str = 'the movie'
seed_as_id = seed_gen(seed_str, word2id=word2id)

max_len = 40

# Generate the review:
generate_review(model, max_len, id2word, seed=seed_as_id, sent="negative", mix_ind=None)
# generate_review(model, max_len, id2word, seed=[1], sent="positive")
