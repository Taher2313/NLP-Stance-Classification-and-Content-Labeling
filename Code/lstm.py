from copy import deepcopy
import numpy as np
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from keras.metrics import AUC
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from imblearn.over_sampling import SMOTE


class LSTM:
    def __init__(self, embeddings , train_x , dev_x , train_label , dev_label , output_size=3, learning_rate=0.0001 , label=1 , path='model/' , epochs=4):
        self.embeddings = embeddings
        self.train_x = train_x
        self.dev_x = dev_x
        self.train_label = train_label
        self.dev_label = dev_label
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.label = label
        self.path = path
        self.epochs = epochs

    def tweets_to_word_vectors(self , x):
        vectors = []
        for sentence in x:
            sentence_vector = []
            for token in sentence:
                if token not in self.embeddings:
                    continue
                token_vector = self.embeddings[token]
                sentence_vector.append(token_vector)
            vectors.append(sentence_vector)
        return vectors

    def pad_X(self , X, desired_sequence_length):
        X_copy = deepcopy(X)
        for i, x in enumerate(X):
            sequence_length_difference = desired_sequence_length - len(x)
            pad = np.zeros(shape=(sequence_length_difference, 300))
            X_copy[i] = np.concatenate([x, pad])
        return np.array(X_copy).astype(float)

    def build_model(self , max_length):
        model = Sequential([])
        model.add(layers.Input(shape=(max_length, 300)))
        model.add(layers.LSTM(128, return_sequences=True))
        model.add(layers.Dropout(0.5))
        model.add(layers.LSTM(128, return_sequences=True))
        model.add(layers.Dropout(0.5))
        model.add(layers.LSTM(128, return_sequences=False))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.output_size, activation='softmax'))
        cp = ModelCheckpoint(self.path, save_best_only=True)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                    loss= SparseCategoricalCrossentropy(),
                    metrics=['accuracy'])

        return model , cp

    def train(self):
        self.train_x = self.tweets_to_word_vectors(self.train_x)
        self.dev_x = self.tweets_to_word_vectors(self.dev_x)
        max_length = max([len(x) for x in self.train_x])
        self.train_x = self.pad_X(self.train_x, max_length)
        self.dev_x = self.pad_X(self.dev_x, max_length)
        self.oversample()
        model , cp = self.build_model(max_length)
        model.fit(self.train_x, self.train_label, validation_data=(self.dev_x, self.dev_label), epochs=self.epochs, callbacks=[cp])
        
    def oversample(self):
        temp = np.empty((len(self.train_x),101*300))
        for i in range(len(self.train_x)):
            temp[i] = self.train_x[i].flatten()

        smote = SMOTE(random_state=42)

        self.train_x = temp
        self.train_x , self.train_label = smote.fit_resample(self.train_x, self.train_label)
        self.train_x = self.train_x.reshape((len(self.train_x),101,300))

    def get_predictions(self):
        self.train()
        best_model = load_model(self.path)
        test_predictions =  best_model.predict(self.dev_x)
        predictions = np.argmax(test_predictions, axis = 1)

        if self.label == 1:
            predictions = predictions - 1

        return predictions
