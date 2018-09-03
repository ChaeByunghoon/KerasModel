from FileController import FileManager
from CNNModel import CNNModel
from keras.preprocessing.text import Tokenizer


fm = FileManager("/Users/")

# parameter
#self, num_words=None,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=' ',char_level=False, oov_token=None,
t = Tokenizer(num_words=3)
seq = t.texts_to_sequences(fm.all_vector_list)


cnn = CNNModel(fm.all_node_list, fm.answer_list)

model = cnn.build_model()

model.fit(seq, fm.answer_list, epochs=10)
