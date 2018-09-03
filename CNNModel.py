from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Embedding

from keras.utils import np_utils
import numpy as np


class CNNModel:

    def __init__(self, input_data, answer_data):
        self._input_data = input_data
        self._answer_data = answer_data

    def build_model(self):
        self.model = Sequential()
        # Embedding model
        self.model.add(Embedding(input_dim=None, output_dim=50))
        self.model.add(Dropout(0.2))
        self.model.add(Conv1D(kernel_size=64, strides=5, filters=10, activation='relu', use_bias=True))
        self.model.add(MaxPooling1D(pool_size=4))
        self.model.add(Dense(1, activation='sigmoid'))
        # or sgd
        # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return self.model
