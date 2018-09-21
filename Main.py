from FileController import FileManager
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import flag
from CNNModel import CNNModel
from EmbeddingModel import EmbeddingModel
import numpy as np
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 파일매니저
fm = FileManager(flag.data_path)
# 인풋 쉐이프

# x_train, y_train, x_test, y_test = fm.get_train_and_validation_set()
node_lists, answer_list = fm.get_node_answer_list()
node_lists = fm.except_infrequent_word(node_lists)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(node_lists)
seq = tokenizer.texts_to_sequences(node_lists)

x = sequence.pad_sequences(seq, maxlen=flag.sequence_length, padding="post", truncating="post")


x_train, y_train, x_test, y_test = fm.get_train_test_cv_set(1, x, answer_list)

word_index = tokenizer.word_index


model_generator = CNNModel(word_index=word_index)
cnn_model = model_generator.build_model()

cnn_model.fit(x_train, y_train, epochs=15, batch_size=16, validation_data=(x_test, y_test))

