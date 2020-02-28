import tensorflow as tf
import numpy as np
import pandas as pd

import os

#TRAIN_DATA_URL = "https://drive.google.com/uc?export=view&id=1QMikptOsJE0IWYIsTgyCfQS9z7hwgle5"
#TRAIN_DATA = tf.keras.utils.get_file("myPersonality_Training.csv", TRAIN_DATA_URL)
TRAIN_DATA = "mypersonality_binary.csv"
TEST_DATA =  "mypersonality_test.csv"
#using a google drive link here
#GLOVE_EMBEDDING_URL = "https://drive.google.com/uc?export=download&confirm=d15I&id=1ipP0FHT6AFWEP_FRMxSAOXj-OQr_dhAz"
GLOVE_EMBEDDING = "glove.6B.100d.txt"

#Prepare the dataset
train = pd.read_csv(TRAIN_DATA)
test = pd.read_csv(TEST_DATA)
test_ln = 99

train["STATUS"].fillna("fillna")
test["STATUS"].fillna("fillna")

labels = ["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]
x_train = train["STATUS"].str.lower()
y_train = train[labels].values

x_test = test["STATUS"].str.lower()
y_test = test[labels].values

max_words = 10433
max_len = 150

embed_size = 100

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words, lower=True)

tokenizer.fit_on_texts(x_train)

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)


checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model = tf.keras.models.load_model('model1.h5')
latest = tf.train.latest_checkpoint(checkpoint_dir)

#Predictions from the model
def prediction(number):
    model.load_weights(latest)
    predictions = model.predict(np.expand_dims(x_train[number], 0))

    print(tokenizer.sequences_to_texts([x_train[number]]))
    print(y_train[number])
    print(labels)
    print(predictions)

#Predictions from user input
def input_prediction():
    input1 = pd.Series(["just watching a movie, waiting for my adventure to begin!"],dtype='string')
    input1 = input1.str.lower()
    token = tokenizer.texts_to_sequences(input1)
    token = tf.keras.preprocessing.sequence.pad_sequences(token, maxlen=max_len)
    model.load_weights(latest)
    #Figured out the issue, have to make a pandas series, then only present the first one
    predictions = model.predict(np.expand_dims(token[0],0))

    print(tokenizer.sequences_to_texts(token[0]))
    print(labels)
    print(predictions)

def testing_prediction():
    model.load_weights(latest)
    number = 0
    while number < test_ln:
        #or do a while loop :(
        predictions = model.predict(np.expand_dims(x_test[number],0))

        print(tokenizer.sequences_to_texts([x_test[number]]))
        print(y_test[number])
        print(labels)
        print(predictions)
        print("")
        number += 1

prediction(43)
prediction(55)
#input_prediction()
#This prints a lot of text, comment out if you are concerned about that
#testing_prediction()
