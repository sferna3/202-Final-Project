import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

#TRAIN_DATA_URL = "https://drive.google.com/uc?export=view&id=1QMikptOsJE0IWYIsTgyCfQS9z7hwgle5"
#TRAIN_DATA = tf.keras.utils.get_file("myPersonality_Training.csv", TRAIN_DATA_URL)
TRAIN_DATA = "mypersonality_binary.csv"
#using a google drive link here
#GLOVE_EMBEDDING_URL = "https://drive.google.com/uc?export=download&confirm=d15I&id=1ipP0FHT6AFWEP_FRMxSAOXj-OQr_dhAz"
GLOVE_EMBEDDING = "glove.6B.100d.txt"

#Prepare the dataset
train = pd.read_csv(TRAIN_DATA)

train["STATUS"].fillna("fillna")
labels = ["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]
x_train = train["STATUS"].str.lower()
y_train = train[labels].values

max_words = 10433
max_len = 150
epochs = 40

embed_size = 100

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words, lower=True)

tokenizer.fit_on_texts(x_train)

x_train = tokenizer.texts_to_sequences(x_train)

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)


#Using pre-trained embeddings (from file)
embeddings_index = {}

with open(GLOVE_EMBEDDING, encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        embed = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embed

word_index = tokenizer.word_index

num_words = min(max_words, len(word_index) + 1)

embedding_matrix = np.zeros((num_words, embed_size), dtype='float32')

for word, i in word_index.items():

    if i >= max_words:
        continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#Embedding matrix
input = tf.keras.layers.Input(shape=(max_len,))

x = tf.keras.layers.Embedding(max_words, embed_size, weights=[embedding_matrix], trainable=False)(input)

#Bidirectional layer
x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, dropout=0.1,
                                                      recurrent_dropout=0.1))(x)

x = tf.keras.layers.Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(x)

avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)

x = tf.keras.layers.concatenate([avg_pool, max_pool])

#MAKE SURE LAYERS HERE MATCH NUMBER OF CLASSES!!!!!
preds = tf.keras.layers.Dense(5, activation="sigmoid")(x)

model = tf.keras.Model(input, preds)

model.summary()


#Training the model

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=1e-3), metrics=['accuracy'])
batch_size = 128
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
callbacks = [
#got an error here, switching to "loss"
    #tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='loss'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    cp_callback
    ]

history = model.fit(x_train, y_train, validation_split=0.2, batch_size=batch_size,
        epochs=epochs, callbacks=callbacks, verbose=1)

model.save("model1.h5")

#Creating a graph of accuracy and loss over time
history_dict = history.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
