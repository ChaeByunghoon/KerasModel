from FileController import FileManager
from CNNModel import CNNModel
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers import Input
import flag

# 파라미터
embedding_dim = 30
batch_size = 16

# 파일매니저
fm = FileManager("/Users/chaebyeonghun/Desktop/IEEEData_Final/camel/")
# 인풋 쉐이프

# x_train, y_train, x_test, y_test = fm.get_train_and_validation_set()
node_lists, answer_list = fm.get_node_answer_list()

tokenizer = Tokenizer(num_words=3)
tokenizer.fit_on_texts(node_lists)
seq = tokenizer.texts_to_sequences(node_lists)

# sequence
x = sequence.pad_sequences(seq, maxlen=flag.sequence_length, padding="post", truncating="post")

x_train, y_train, x_test, y_test = fm.get_train_test_set(x, answer_list)

word_index = tokenizer.word_index

print("Found %s unique tokens" % len(word_index))


model_generator = CNNModel(word_index=word_index)
cnn_model = model_generator.build_model()

# epoch 학습용 데이터 셋을 몇번 돌릴 것인지?
# batch_size 한번에 학습할때 얼만큼 데이터를 공급할 것인지,

cnn_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))



