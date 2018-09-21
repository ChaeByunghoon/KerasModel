from FileController import FileManager
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import flag
from CNNModel import CNNModel
import numpy as np
# 파일매니저
fm = FileManager(flag.data_path)
# 인풋 쉐이프

node_lists, answer_list = fm.get_node_answer_list()
node_lists = fm.except_infrequent_word(node_lists)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(node_lists)
seq = tokenizer.texts_to_sequences(node_lists)

# sequence
x = sequence.pad_sequences(seq, maxlen=flag.sequence_length)

for i in range(1, 11):
    y = np.array(answer_list)
    y = y.reshape(len(y), 1)
    x_train, y_train, x_test, y_test = fm.get_train_test_cv_set(i, x, answer_list)

    word_index = tokenizer.word_index

    print(flag.data_path, i, "th Cross Validation")

    model_generator = CNNModel(word_index=word_index)
    cnn_model = model_generator.build_model()

    cnn_model.fit(x, y, epochs=15, batch_size=32, validation_data=(x_test, y_test))

