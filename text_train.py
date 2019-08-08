import urllib.request
import os
import tarfile
import re
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from keras.layers import Bidirectional
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import LSTM

data_path = r"data/aclImdb/"


# get IMDb data and unzip it
def get_imdb_data_and_unzip():
    url = r"http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    imdb_file_path = "data/aclImdb_v1.tar.gz"

    print("start to download data")
    if not os.path.isfile(imdb_file_path):
        if not os.path.exists(data_path):
            result = urllib.request.urlretrieve(url, imdb_file_path)
            print("downloaded:", result)
        else:
            print("find aclImdb folder, already download done, skip download step")
    else:
        print("already download done, skip download step")

    print("start to unzip data")
    if not os.path.exists(data_path):
        t_file = tarfile.open(imdb_file_path, 'r:gz')
        t_file.extractall('data/')
        print("unzip done")
    else:
        print("already unzip done, skip unzip step")


# remove HTML tag
def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)


def read_files(filetype):
    file_list = []

    positive_path = data_path + filetype + "/pos/"
    for f in os.listdir(positive_path):
        file_list += [positive_path + f]

    negative_path = data_path + filetype + "/neg/"
    for f in os.listdir(negative_path):
        file_list += [negative_path + f]

    print('read', filetype, 'file:', len(file_list))

    all_labels = ([1] * 12500 + [0] * 12500)
    all_texts = []
    for fi in file_list:
        with open(fi, encoding='utf8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]
    return all_labels, all_texts


def pre_process_data():
    y_train, train_text = read_files("train")
    y_test, test_text = read_files("test")
    # print(train_text[12501])
    # print(y_train[12501])

    token = Tokenizer(num_words=2000)
    token.fit_on_texts(train_text)
    # print(token.document_count)
    # print(token.word_index)

    x_train_seq = token.texts_to_sequences(train_text)
    x_test_seq = token.texts_to_sequences(test_text)
    # print(train_text[0])
    # print(x_train_seq[0])
    x_train = sequence.pad_sequences(x_train_seq, maxlen=100)
    x_test = sequence.pad_sequences(x_test_seq, maxlen=100)
    # print("before pad_sequences len=", len(x_train_seq[0]))
    # print("after pad_sequences len=", len(x_train[0]))


def create_model(model_type):
    model = Sequential()
    model.add(Embedding(output_dim=32,
                        input_dim=3800,
                        input_length=380))
    model.add(Dropout(0.2))

    if model_type == 'BRNN' or model_type == 'DBRNN':
        model.add(Bidirectional(SimpleRNN(units=16, return_sequences=True), merge_mode='concat'))
    elif model_type == 'RNN':
        model.add(SimpleRNN(units=16))
    elif model_type == 'LSTM':
        model.add(LSTM(32))

    if model_type == 'DBRNN':
        model.add(SimpleRNN(units=8))

    if model_type == 'BRNN' or model_type == 'MLP':
        model.add(Flatten())

    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(units=1, activation='sigmoid'))

    return model


def start_train():
    y_train, train_text = read_files("train")
    y_test, test_text = read_files("test")
    # print(train_text[12501])
    # print(y_train[12501])

    token = Tokenizer(num_words=3800)
    token.fit_on_texts(train_text)
    # print(token.document_count)
    # print(token.word_index)

    x_train_seq = token.texts_to_sequences(train_text)
    x_test_seq = token.texts_to_sequences(test_text)
    # print(train_text[0])
    # print(x_train_seq[0])
    x_train = sequence.pad_sequences(x_train_seq, maxlen=380)
    x_test = sequence.pad_sequences(x_test_seq, maxlen=380)
    # print("before pad_sequences len=", len(x_train_seq[0]))
    # print("after pad_sequences len=", len(x_train[0]))

    # model = create_model('MLP')
    # model = create_model('RNN')
    model = create_model('BRNN')
    # model = create_model('DBRNN')
    # model = create_model('LSTM')

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    es = EarlyStopping(monitor='val_acc', patience=5, verbose=0, mode='auto')
    model.fit(x_train, y_train,
              batch_size=100,
              epochs=10,
              verbose=2,
              validation_split=0.2,
              # callbacks=[es],
              shuffle=True)

    scores = model.evaluate(x_test, y_test, verbose=1)
    print(scores[1])


if __name__ == '__main__':
    get_imdb_data_and_unzip()
    start_train()

